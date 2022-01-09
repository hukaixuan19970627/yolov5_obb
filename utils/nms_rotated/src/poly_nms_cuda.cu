#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCDeviceUtils.cuh>

#include <vector>
#include <iostream>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;


#define maxn 10
const double eps=1E-8;

__device__ inline int sig(float d){
    return(d>eps)-(d<-eps);
}

__device__ inline int point_eq(const float2 a, const float2 b) {
    return sig(a.x - b.x) == 0 && sig(a.y - b.y)==0;
}

__device__ inline void point_swap(float2 *a, float2 *b) {
    float2 temp = *a;
    *a = *b;
    *b = temp;
}

__device__ inline void point_reverse(float2 *first, float2* last)
{
    while ((first!=last)&&(first!=--last)) {
        point_swap (first,last);
        ++first;
    }
}

__device__ inline float cross(float2 o,float2 a,float2 b){  //叉积
    return(a.x-o.x)*(b.y-o.y)-(b.x-o.x)*(a.y-o.y);
}
__device__ inline float area(float2* ps,int n){
    ps[n]=ps[0];
    float res=0;
    for(int i=0;i<n;i++){
        res+=ps[i].x*ps[i+1].y-ps[i].y*ps[i+1].x;
    }
    return res/2.0;
}
__device__ inline int lineCross(float2 a,float2 b,float2 c,float2 d,float2&p){
    float s1,s2;
    s1=cross(a,b,c);
    s2=cross(a,b,d);
    if(sig(s1)==0&&sig(s2)==0) return 2;
    if(sig(s2-s1)==0) return 0;
    p.x=(c.x*s2-d.x*s1)/(s2-s1);
    p.y=(c.y*s2-d.y*s1)/(s2-s1);
    return 1;
}

__device__ inline void polygon_cut(float2*p,int&n,float2 a,float2 b, float2* pp){

    int m=0;p[n]=p[0];
    for(int i=0;i<n;i++){
        if(sig(cross(a,b,p[i]))>0) pp[m++]=p[i];
        if(sig(cross(a,b,p[i]))!=sig(cross(a,b,p[i+1])))
            lineCross(a,b,p[i],p[i+1],pp[m++]);
    }
    n=0;
    for(int i=0;i<m;i++)
        if(!i||!(point_eq(pp[i], pp[i-1])))
            p[n++]=pp[i];
    // while(n>1&&p[n-1]==p[0])n--;
    while(n>1&&point_eq(p[n-1], p[0]))n--;
}

//---------------华丽的分隔线-----------------//
//返回三角形oab和三角形ocd的有向交面积,o是原点//
__device__ inline float intersectArea(float2 a,float2 b,float2 c,float2 d){
    float2 o = make_float2(0,0);
    int s1=sig(cross(o,a,b));
    int s2=sig(cross(o,c,d));
    if(s1==0||s2==0)return 0.0;//退化，面积为0
    // if(s1==-1) swap(a,b);
    // if(s2==-1) swap(c,d);
    if (s1 == -1) point_swap(&a, &b);
    if (s2 == -1) point_swap(&c, &d);
    float2 p[10]={o,a,b};
    int n=3;
    float2 pp[maxn];
    polygon_cut(p,n,o,c,pp);
    polygon_cut(p,n,c,d,pp);
    polygon_cut(p,n,d,o,pp);
    float res=fabs(area(p,n));
    if(s1*s2==-1) res=-res;return res;
}
//求两多边形的交面积
__device__ inline float intersectArea(float2*ps1,int n1,float2*ps2,int n2){
    if(area(ps1,n1)<0) point_reverse(ps1,ps1+n1);
    if(area(ps2,n2)<0) point_reverse(ps2,ps2+n2);
    ps1[n1]=ps1[0];
    ps2[n2]=ps2[0];
    float res=0;
    for(int i=0;i<n1;i++){
        for(int j=0;j<n2;j++){
            res+=intersectArea(ps1[i],ps1[i+1],ps2[j],ps2[j+1]);
        }
    }
    return res;//assumeresispositive!
}

// TODO: optimal if by first calculate the iou between two hbbs
__device__ inline float devPolyIoU(float const * const p, float const * const q) {
    float2 ps1[maxn], ps2[maxn];
    int n1 = 4;
    int n2 = 4;
    for (int i = 0; i < 4; i++) {
        ps1[i].x = p[i * 2];
        ps1[i].y = p[i * 2 + 1];

        ps2[i].x = q[i * 2];
        ps2[i].y = q[i * 2 + 1];
    }
    float inter_area = intersectArea(ps1, n1, ps2, n2);
    float union_area = fabs(area(ps1, n1)) + fabs(area(ps2, n2)) - inter_area;
    float iou = 0;
    if (union_area == 0) {
        iou = (inter_area + 1) / (union_area + 1);
    } else {
        iou = inter_area / union_area;
    }
    return iou;
}

__global__ void poly_nms_kernel(const int n_polys, const float nms_overlap_thresh,
                            const float *dev_polys, unsigned long long *dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size =
            min(n_polys - row_start * threadsPerBlock, threadsPerBlock);
    const int cols_size =
            min(n_polys - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_polys[threadsPerBlock * 9];
    if (threadIdx.x < cols_size) {
        block_polys[threadIdx.x * 9 + 0] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 0];
        block_polys[threadIdx.x * 9 + 1] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 1];
        block_polys[threadIdx.x * 9 + 2] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 2];
        block_polys[threadIdx.x * 9 + 3] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 3];
        block_polys[threadIdx.x * 9 + 4] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 4];
        block_polys[threadIdx.x * 9 + 5] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 5];
        block_polys[threadIdx.x * 9 + 6] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 6];
        block_polys[threadIdx.x * 9 + 7] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 7];
        block_polys[threadIdx.x * 9 + 8] =
            dev_polys[(threadsPerBlock * col_start + threadIdx.x) * 9 + 8];
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        const float *cur_box = dev_polys + cur_box_idx * 9;
        int i = 0;
        unsigned long long t = 0;
        int start = 0;
        if (row_start == col_start) {
            start = threadIdx.x + 1;
        }
        for (i = start; i < cols_size; i++) {
            if (devPolyIoU(cur_box, block_polys + i * 9) > nms_overlap_thresh) {
                t |= 1ULL << i;
            }
        }
        const int col_blocks = THCCeilDiv(n_polys, threadsPerBlock);
        dev_mask[cur_box_idx * col_blocks + col_start] = t;
    }
}

// boxes is a N x 9 tensor
at::Tensor poly_nms_cuda(const at::Tensor boxes, float nms_overlap_thresh) {

    at::DeviceGuard guard(boxes.device());

    using scalar_t = float;
    AT_ASSERTM(boxes.device().is_cuda(), "boxes must be a CUDA tensor");
    auto scores = boxes.select(1, 8);
    auto order_t = std::get<1>(scores.sort(0, /*descending=*/true));
    auto boxes_sorted = boxes.index_select(0, order_t);

    int boxes_num = boxes.size(0);

    const int col_blocks = THCCeilDiv(boxes_num, threadsPerBlock);

    scalar_t* boxes_dev = boxes_sorted.data_ptr<scalar_t>();

    THCState *state = at::globalContext().lazyInitCUDA();

    unsigned long long* mask_dev = NULL;

    mask_dev = (unsigned long long*) THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));

    dim3 blocks(THCCeilDiv(boxes_num, threadsPerBlock),
                THCCeilDiv(boxes_num, threadsPerBlock));
    dim3 threads(threadsPerBlock);
    poly_nms_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(boxes_num,
                                        nms_overlap_thresh,
                                        boxes_dev,
                                        mask_dev);
    
    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    THCudaCheck(cudaMemcpyAsync(
			    &mask_host[0],
                            mask_dev,
                            sizeof(unsigned long long) * boxes_num * col_blocks,
                            cudaMemcpyDeviceToHost,
			    at::cuda::getCurrentCUDAStream()
			    ));
    
    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    at::Tensor keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong).device(at::kCPU));
    int64_t* keep_out = keep.data_ptr<int64_t>();

    int num_to_keep = 0;
    for (int i = 0; i < boxes_num; i++) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_out[num_to_keep++] = i;
            unsigned long long *p = &mask_host[0] + i * col_blocks;
            for (int j = nblock; j < col_blocks; j++) {
                remv[j] |= p[j];
            }
        }
    }

    THCudaFree(state, mask_dev);

    return order_t.index({
        keep.narrow(/*dim=*/0, /*start=*/0, /*length=*/num_to_keep).to(
          order_t.device(), keep.scalar_type())});
}

