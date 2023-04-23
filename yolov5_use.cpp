#include <iostream>
#include <chrono>
#include <cmath>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include<string>
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.25
#define CONF_THRESH 0.25
#define objThreshold 0.25
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !
using namespace std;
// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const bool keep_ratio = true;
static Logger gLogger;
using namespace cv;
using namespace std;
typedef struct BoxInfo
{
	RotatedRect box;
	float score;
	int label;
} BoxInfo;

cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = INPUT_H;
	*neww = INPUT_W;
	cv::Mat dstimg;
	if (keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = INPUT_H;
			*neww = INPUT_W;
			cv::resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((INPUT_W - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, INPUT_W - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)INPUT_H * hw_scale;
			*neww = INPUT_W;
			cv::resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(INPUT_H - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, INPUT_H - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;


}
void nms_angle(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = input_boxes[i].box.size.area();
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			vector<Point2f> intersectingRegion;
			rotatedRectangleIntersection(input_boxes[i].box, input_boxes[j].box, intersectingRegion);
			if (intersectingRegion.empty()) { continue; }
			float inter = contourArea(intersectingRegion);
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= NMS_THRESH)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize) {
	CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));// 把这个任务通过cudaMemcpyAsync放到cudastram上

	// infer on the batch asynchronously, and DMA output back to host
	context.enqueue(batchSize, buffers, stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}

int main(int argc, char** argv) {
	cudaSetDevice(DEVICE);

	std::string engine_name = "D:\\workshop\\yolo-obb\\yolov5-obb-tensorrt-infer-main\\obb-tensorrt\\yolov5_obb_tensorrt_cpp\\build\\yolov5pro.engine";


	// deserialize the .engine and run inference
	std::ifstream file(engine_name, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << engine_name << " error!" << std::endl;
		return -1;
	}
	char *trtModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtModelStream = new char[size];
	assert(trtModelStream);
	file.read(trtModelStream, size);
	file.close();

	std::vector<std::string> file_names;
	//file_names.push_back("D:\\workshop\\yolo-obb\\CTdata\\newdata\\images\\BGCT-0824_20200617_1234_20230202090406_000001-x-090.png");
	file_names.push_back("D:\\workshop\\yolo-obb\\CTdata\\png\\png\\BGCT-0824_20200617_1234_20230202090406_000001-x-090.png");
	static float prob[BATCH_SIZE * OUTPUT_SIZE];
	IRuntime* runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	assert(engine->getNbBindings() == 2);
	float* buffers[2];
	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	assert(inputIndex == 0);
	assert(outputIndex == 1);
	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc((void**)&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	uint8_t* img_host = nullptr;
	uint8_t* img_device = nullptr;
	// prepare input data cache in pinned memory 
	CUDA_CHECK(cudaMallocHost((void**)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
	// prepare input data cache in device memory
	CUDA_CHECK(cudaMalloc((void**)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
	static float data[1 * 3 * INPUT_H * INPUT_W];
	

	cv::Mat pr_img = cv::imread(file_names[0]);
	//cv::resize(pr_img, pr_img, Size(512, 512), INTER_AREA);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat img = resize_image(pr_img, &newh, &neww, &padh, &padw);
	//cv::imshow("pr_img.jpg", pr_img);
	int i = 0;
	int b = 0;//这个b原本用来控制batchsize的，但是我们只有一张图片 就设置成了0
	for (int row = 0; row < INPUT_H; ++row)
	{
		uchar* uc_pixel = img.data + row * img.step;
		for (int col = 0; col < INPUT_W; ++col)
		{
			data[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
			data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

	// Run inference
	//for (int i = 0; i < 10; i++) {
	//	auto start = std::chrono::system_clock::now();
	//	doInference(*context, stream, (void**)buffers, data, prob, 1);
	//	auto end = std::chrono::system_clock::now();
	//	std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	//}
	auto start = std::chrono::system_clock::now();
	doInference(*context, stream, (void**)buffers, data, prob, 1);
	vector<BoxInfo> generate_boxes;
	float ratioh = (float)pr_img.rows / newh, ratiow = (float)pr_img.cols / neww;
	float* pdata = (float*)prob;
	int det_size = 7;//box(4)+conf +classid + classangle
	for (int i = 0; i < pdata[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
		
		if (pdata[7 * i + 4+1] <= CONF_THRESH) continue;//这里你会很奇怪为什么从pdata[5]是conf 因为pdata[0]存放的是0，代表第一个图片
		Yolo::Detection det;
		memcpy(&det, &pdata[1 + det_size * i], det_size * sizeof(float));
		float max_class_socre = det.conf;//置信度
		if (max_class_socre > CONF_THRESH)
		{
			float cx = (det.bbox[0] - padw)*ratiow;
			float cy = (det.bbox[1] - padh)*ratioh;
			float w = det.bbox[2] * ratiow;
			float h = det.bbox[3] * ratioh;

			int class_idx =det.class_id ;
			float angle = 90 - det.angle_id;
			RotatedRect box = RotatedRect(Point2f(cx, cy), Size2f(w, h), angle);

			generate_boxes.push_back(BoxInfo{ box, (float)max_class_socre, class_idx });
		}

	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms_angle(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		RotatedRect rectInput = generate_boxes[i].box;
		Point2f* vertices = new cv::Point2f[4];
		rectInput.points(vertices);
		for (int j = 0; j < 4; j++)
		{
			line(pr_img, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 2);
		}

		int xmin = (int)vertices[0].x;
		int ymin = (int)vertices[0].y - 10;
		string label = format("%.2f", generate_boxes[i].score);
		label = format("%d", generate_boxes[i].label) + ":" + label;
		putText(pr_img, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}

	auto end = std::chrono::system_clock::now();
	std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
	namedWindow("result.jpg", 2);
	cv::imshow("result.jpg", pr_img);
	cv::imwrite("result.jpg", pr_img);
	cv::waitKey(0);


	// Release stream and buffers
	cudaStreamDestroy(stream);
	CUDA_CHECK(cudaFree(img_device));
	CUDA_CHECK(cudaFreeHost(img_host));
	CUDA_CHECK(cudaFree(buffers[inputIndex]));
	CUDA_CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	return 0;
}
