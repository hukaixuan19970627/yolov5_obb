import glob
import math
import os
import random
import shutil
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, torch_distributed_zero_first, cvminAreaRect2longsideformat, longsideformat2cvminAreaRect

help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8):
    '''
    确保只有DDP中的第一个进程首先处理数据集，然后其他进程可以使用缓存。
    Make sure only the first process in DDP process the dataset first, and the following others can use the cache.
    return:
        dataloader ： 数据加载器，结合了数据集和取样器
            i: batch_index, 第i个batch (索引方式)  以下为具体数据加载器中的内容
            imgs : torch.Size([batch_size, 3, resized_noheight, resized_width])
            targets : torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, 经归一化后的xywh])
            paths : List['img1_path','img2_path',......,'img-1_path']  len(paths)=batch_size
            shapes ： size= batch_size, 不进行mosaic时进行矩形训练时才有值
        Class dataset 其中有:
            self.img_files  路径文件夹下所有图片路径   self.img_files=['??\\images\\train2017\\1.jpg',...,]
            self.label_files  路径文件夹下所有label_txt路径   self.label_files=['??\\labels\\train2017\\1.txt',...,]
            self.n          路径文件夹下所有图片的总数量
            self.batch , self.img_size , self.augment , self.hyp , self.image_weights , self.rect , self.mosaic , self.mosaic_border , self.stride ,
            self.shapes     [[1.jpg的形状]...[n.jpg的形状]]    eg：[[480 80][360 640]...[480 640]]
            self.labels     [array( [对应1.txt的labels信息] ，dtype=float32), ..., array( [对应n.txt的labels信息] ，dtype=float32)]
    '''
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path, imgsz, batch_size,
                                      augment=augment,  # augment images
                                      hyp=hyp,  # augmentation hyperparameters
                                      rect=rect,  # rectangular training
                                      cache_images=cache,
                                      single_cls=opt.single_cls,
                                      stride=int(stride),
                                      pad=pad,
                                      rank=rank)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if rank != -1 else None
    dataloader = InfiniteDataLoader(dataset,
                                    # 从数据库中每次抽出batch size个样本
                                    batch_size=batch_size,
                                    num_workers=nw,
                                    sampler=sampler,
                                    pin_memory=True,
                                    collate_fn=LoadImagesAndLabels.collate_fn)  # torch.utils.data.DataLoader()
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    Dataloader that reuses workers.
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:  # for inference
    '''
    for inference. LoadImages(path, img_size=640)

    '''
    def __init__(self, path, img_size=640):
        p = str(Path(path))  # os-agnostic
        p = os.path.abspath(p)  # absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception('ERROR: %s does not exist' % p)

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'images'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, 'No images or videos found in %s. Supported formats are:\nimages: %s\nvideos: %s' % \
                            (p, img_formats, vid_formats)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        '''
        return path, img,         img0, self.cap
        返回路径，resize+pad的图片，原始图片，视频对象
        '''
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print('video %g/%g (%g/%g) %s: ' % (self.count + 1, self.nf, self.frame, self.nframes, path), end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print('image %g/%g %s: ' % (self.count, self.nf, path), end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        # 返回路径，resize+pad的图片，原始图片，视频对象
        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


class LoadWebcam:  # for inference
    def __init__(self, pipe=0, img_size=640):
        self.img_size = img_size

        if pipe == '0':
            pipe = 0  # local camera
        # pipe = 'rtsp://192.168.1.64/1'  # IP camera
        # pipe = 'rtsp://username:password@192.168.1.64/1'  # IP camera with login
        # pipe = 'rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa'  # IP traffic camera
        # pipe = 'http://wmccpinetop.axiscam.net/mjpg/video.mjpg'  # IP golf camera

        # https://answers.opencv.org/question/215996/changing-gstreamer-pipeline-to-opencv-in-pythonsolved/
        # pipe = '"rtspsrc location="rtsp://username:password@192.168.1.64/1" latency=10 ! appsink'  # GStreamer

        # https://answers.opencv.org/question/200787/video-acceleration-gstremer-pipeline-in-videocapture/
        # https://stackoverflow.com/questions/54095699/install-gstreamer-support-for-opencv-python-package  # install help
        # pipe = "rtspsrc location=rtsp://root:root@192.168.0.91:554/axis-media/media.amp?videocodec=h264&resolution=3840x2160 protocols=GST_RTSP_LOWER_TRANS_TCP ! rtph264depay ! queue ! vaapih264dec ! videoconvert ! appsink"  # GStreamer

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)  # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # set buffer size

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):  # q to quit
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if self.pipe == 0:  # local camera
            ret_val, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)  # flip left-right
        else:  # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if n % 30 == 0:  # skip frames
                    ret_val, img0 = self.cap.retrieve()
                    if ret_val:
                        break

        # Print
        assert ret_val, 'Camera Error %s' % self.pipe
        img_path = 'webcam.jpg'
        print('webcam %g: ' % self.count, end='')

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 4:  # read every 4th frame
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.01)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

def rotate_augment(angle, scale, image, labels):
    """
    旋转目标增强  随机旋转
    @param angle: 旋转增强角度 int 单位为度
    @param scale: 设为1,尺度由train.py中定义
    @param image:  img信息  shape(heght, width, 3)
    @param labels:  (num, [classid x_c y_c longside shortside Θ]) Θ ∈ int[0,180)
    @return:
           array rotated_img: augmented_img信息  shape(heght, width, 3)
           array rotated_labels: augmented_label:  (num, [classid x_c y_c longside shortside Θ])
    """
    Pi_angle = -angle * math.pi / 180.0  # 弧度制，后面旋转坐标需要用到，注意负号！！！
    rows, cols = image.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D(center=(a, b), angle=angle, scale=scale)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))  # 旋转后的图像保持大小不变
    rotated_labels = []
    for label in labels:
        # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
        rect = longsideformat2cvminAreaRect(label[1], label[2], label[3], label[4], (label[5] - 179.9))
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)  # 返回rect对应的四个点的值 normalized

        # 四点坐标反归一化
        poly[:, 0] = poly[:, 0] * cols
        poly[:, 1] = poly[:, 1] * rows

        # 下面是计算旋转后目标相对旋转过后的图像的位置
        X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
        Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

        X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
        Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

        X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
        Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

        X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
        Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

        poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])
        # 四点坐标归一化
        poly_rotated[:, 0] = poly_rotated[:, 0] / cols
        poly_rotated[:, 1] = poly_rotated[:, 1] / rows

        rect_rotated = cv2.minAreaRect(np.float32(poly_rotated))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

        c_x = rect_rotated[0][0]
        c_y = rect_rotated[0][1]
        w = rect_rotated[1][0]
        h = rect_rotated[1][1]
        theta = rect_rotated[-1]  # Range for angle is [-90，0)
        # (num, [classid x_c y_c longside shortside Θ])
        label[1:] = cvminAreaRect2longsideformat(c_x, c_y, w, h, theta)

        if (sum(label[1:-1] <= 0) + sum(label[1:3] >= 1)) >= 1:  # 0<xy<1, 0<side<=1
            # print('bbox[:2]中有>= 1的元素,bbox中有<= 0的元素,已将某个box排除,')
            np.clip(label[1:-1], 0, 1, out=label[1:-1])

        label[-1] = int(label[-1] + 180.5)  # range int[0,180] 四舍五入
        if label[-1] == 180:  # range int[0,179]
            label[-1] = 179
        rotated_labels.append(label)

    return rotated_img, np.array(rotated_labels)

class LoadImagesAndLabels(Dataset):
    """
    for training/testing
    Args:
        path: train_path or test_path  eg：../coco128/images/train2017/
        img_size，batch_size，augment，hyp，rect，image_weights，cache_images，single_cls，stride，pad，rank
    return:
        class Dataset:
            self.img_files  路径文件夹下所有图片路径   self.img_files=['??\\images\\train2017\\1.jpg',...,]
            self.label_files  路径文件夹下所有label_txt路径   self.label_files=['??\\labels\\train2017\\1.txt',...,]
            self.n          路径文件夹下所有图片的总数量
            self.batch , self.img_size , self.augment , self.hyp , self.image_weights , self.rect , self.mosaic , self.mosaic_border , self.stride ,
            self.shapes     [[1.jpg的形状]...[n.jpg的形状]]    eg：[[480 80][360 640]...[480 640]]
            self.labels     [array( [对应1.txt的labels信息] ，dtype=float32), ..., array( [对应n.txt的labels信息] ，dtype=float32)]
    """
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, rank=-1):
        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = str(Path(p))  # os-agnostic
                # 举例：parent = ‘..\coco128\images’ + '\'
                parent = str(Path(p).parent) + os.sep
                if os.path.isfile(p):  # file
                    with open(p, 'r') as t:
                        t = t.read().splitlines()
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                elif os.path.isdir(p):  # folder
                    f += glob.iglob(p + os.sep + '*.*')
                else:
                    raise Exception('%s does not exist' % p)
            # p路径文件夹下所有图片路径都会存在self.img_files中   self.img_files=['??\\images\\train2017\\1.jpg',...,]
            self.img_files = sorted(
                [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats])
        except Exception as e:
            raise Exception('Error loading data from %s: %s\nSee %s' % (path, e, help_url))

        n = len(self.img_files)
        assert n > 0, 'No images found in %s. See %s' % (path, help_url)
        bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
        nb = bi[-1] + 1  # number of batches

        self.n = n  # number of images
        self.batch = bi  # batch index of image
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect  # load 4 images at a time into a mosaic (only during training)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride

        # Define labels
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # sa=/images/, sb=/labels/   as substrings
        # p路径文件夹下所有label_txt都会存在self.label_files   self.label_files=['??\\labels\\train2017\\1.txt',...,]
        self.label_files = [x.replace(sa, sb, 1).replace(os.path.splitext(x)[-1], '.txt') for x in self.img_files]

        # Check cache
        # 初始化图片与标签，为缓存图片、标签做准备
        '''
        创建缓存文件cache
        List cache： {
                       '??\\images\\train2017\\1.jpg':[array( [对应1.txt的labels信息] ，dtype=float32), (weights, heights))] ,
                       ...
                       '??\\images\\train2017\\n.jpg':[array( [对应n.txt的labels信息] ，dtype=float32), (weights, heights))]
                     }
        '''
        cache_path = str(Path(self.label_files[0]).parent) + '.cache'  # cached labels
        if os.path.isfile(cache_path):
            cache = torch.load(cache_path)  # load
            if cache['hash'] != get_hash(self.label_files + self.img_files):  # dataset changed
                cache = self.cache_labels(cache_path)  # re-cache
        else:
            cache = self.cache_labels(cache_path)  # cache

        # Get labels
        '''
        self.shapes = [[1.jpg的形状]...[n.jpg的形状]]
        self.labels = [array( [对应1.txt的labels信息] ，dtype=float32), ..., array( [对应n.txt的labels信息] ，dtype=float32)]
        '''
        labels, shapes = zip(*[cache[x] for x in self.img_files])
        self.shapes = np.array(shapes, dtype=np.float64)
        self.labels = list(labels)

        # Rectangular Training  https://github.com/ultralytics/yolov3/issues/232
        if self.rect:
            # Sort by aspect ratio  按纵横比的数值从小到大重新进行排序，矩形训练通常以成批处理
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]  # wh
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(np.int) * stride

        # Cache labels
        create_datasubset, extract_bounding_boxes, labels_loaded = False, False, False
        nm, nf, ne, ns, nd = 0, 0, 0, 0, 0  # number missing, found, empty, datasubset, duplicate
        '''
        self.label_files  路径文件夹下所有label_txt路径   self.label_files=['??\\labels\\train2017\\1.txt',...,]
        self.labels = [array( [对应1.txt的labels信息] ,dtype=float32), ..., array( [对应n.txt的labels信息] ,dtype=float32)]
        '''
        pbar = enumerate(self.label_files)
        if rank in [-1, 0]:
            pbar = tqdm(pbar)
        for i, file in pbar:
            l = self.labels[i]  # label  第i张image的labels信息   size = (目标数量, [class, xywh_center(归一化),Θ])
            if l is not None and l.shape[0]:
                # 判断标签是否有6列  [class ,xywh, Θ]
                assert l.shape[1] == 6, '> 6 label columns: %s' % file
                # 判断标签是否全部>=0
                assert (l >= 0).all(), 'negative labels: %s' % file
                # 判断标签坐标x y 是否归一化
                assert (l[:, 1:3] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s' % file
                # 找出标签中重复的坐标
                if np.unique(l, axis=0).shape[0] < l.shape[0]:  # duplicate rows 若有重复目标则nd自增
                    nd += 1  # print('WARNING: duplicate rows in %s' % self.label_files[i])  # duplicate rows
                # 如果数据集只有一个类，设置类别标签为0
                if single_cls:
                    l[:, 0] = 0  # force dataset into single-class mode
                self.labels[i] = l
                nf += 1  # file found

                # Create subdataset (a smaller dataset)
                if create_datasubset and ns < 1E4:
                    if ns == 0:
                        create_folder(path='./datasubset')
                        os.makedirs('./datasubset/images')
                    exclude_classes = 43
                    if exclude_classes not in l[:, 0]:
                        ns += 1
                        # shutil.copy(src=self.img_files[i], dst='./datasubset/images/')  # copy image
                        with open('./datasubset/images.txt', 'a') as f:
                            f.write(self.img_files[i] + '\n')

                # Extract object detection boxes for a second stage classifier
                # 获取目标框与图片，并将框从图片截取下来保存到本地(默认不使用)
                if extract_bounding_boxes:
                    p = Path(self.img_files[i])   #  第i张image的path
                    img = cv2.imread(str(p))
                    h, w = img.shape[:2]
                    for j, x in enumerate(l):    # l ： label  第i张image的labels信息   size = (目标数量, [class, xywh])
                        f = '%s%sclassifier%s%g_%g_%s' % (p.parent.parent, os.sep, os.sep, x[0], j, p.name)
                        if not os.path.exists(Path(f).parent):
                            os.makedirs(Path(f).parent)  # make new output folder

                        # 对归一化的坐标乘以w，h
                        # x.size = [class ,xywh]
                        b = x[1:] * [w, h, w, h]  # box
                        b[2:] = b[2:].max()  # rectangle to square
                        b[2:] = b[2:] * 1.3 + 30  # pad
                        # xywh格式转xyxy
                        b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)

                        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                        assert cv2.imwrite(f, img[b[1]:b[3], b[0]:b[2]]), 'Failure extracting classifier boxes'
            else:
                ne += 1  # print('empty labels for image %s' % self.img_files[i])  # file empty
                # os.system("rm '%s' '%s'" % (self.img_files[i], self.label_files[i]))  # remove

            if rank in [-1, 0]:
                pbar.desc = 'Scanning labels %s (%g found, %g missing, %g empty, %g duplicate, for %g images)' % (
                    cache_path, nf, nm, ne, nd, n)
        if nf == 0:
            s = 'WARNING: No labels found in %s. See %s' % (os.path.dirname(file) + os.sep, help_url)
            print(s)
            assert not augment, '%s. Can not train without labels.' % s

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        # 提前缓存图片到内存中，可以提升训练速度
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            pbar = tqdm(range(len(self.img_files)), desc='Caching images')
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            for i in pbar:  # max 10k images
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = load_image(self, i)  # img, hw_original, hw_resized
                gb += self.imgs[i].nbytes
                pbar.desc = 'Caching images (%.1fGB)' % (gb / 1E9)

    def cache_labels(self, path='labels.cache'):
        '''
        Cache dataset labels, check images and read shapes
        '''
        x = {}  # dict
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for (img, label) in pbar:
            try:
                l = []
                image = Image.open(img)
                image.verify()  # PIL verify
                # _ = io.imread(img)  # skimage verify (from skimage import io)
                shape = exif_size(image)  # image size
                assert (shape[0] > 9) & (shape[1] > 9), 'image size <10 pixels'
                if os.path.isfile(label):
                    with open(label, 'r') as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)  # labels
                if len(l) == 0:  # 当labels文件中内容为空时也要确保shape一致
                    l = np.zeros((0, 6), dtype=np.float32)
                x[img] = [l, shape]
            except Exception as e:
                x[img] = [None, None]
                print('WARNING: %s: %s' % (img, e))

        x['hash'] = get_hash(self.label_files + self.img_files)
        torch.save(x, path)  # save for next time
        return x

    def __len__(self):
        return len(self.img_files)

    # def __iter__(self):
    #     self.count = -1
    #     print('ran dataset iter')
    #     #self.shuffled_vector = np.random.permutation(self.nF) if self.augment else np.arange(self.nF)
    #     return self

    def __getitem__(self, index):  # 只要实例对象（假定为p）做p[i]运算时，就会调用类中的__getitem__方法
        '''
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes
            @param index: dataset类的索引,只要调用实例对象（假定为p）做p[i]运算时，就会调用__getitem__方法
            @return:
                img: 经预处理后的img;size = [3, resized_height, resized_width]
                labels_out :  (目标数量, [0, classid,归一化后的xywh,Θ])
                self.img_files[index] : 图片索引index的文件路径
                shapes：
        '''
        if self.image_weights:
            index = self.indices[index]

        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp['mosaic']
        if mosaic:
            # Load mosaic
            # img4 : size = (3 , size1, size2);
            # labels : size = (单张img4中的目标GT数量, [classid ,LT_x,LT_y,RB_x,RB_y,Θ]);
            img, labels = load_mosaic(self, index)
            shapes = None

            # MixUp https://arxiv.org/pdf/1710.09412.pdf 对mosaic处理后的图片再一次进行随机mixup处理
            if random.random() < hyp['mixup']:
                img2, labels2 = load_mosaic(self, random.randint(0, len(self.labels) - 1))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)

        else:
            # Load image
            # 加载图片并根据设定的输入大小与图片原大小的比例ratio进行resize(未做填充pad到正方形)
            img, (h0, w0), (h, w) = load_image(self, index)

            # Letterbox
            # 如果进行矩形训练，则获取每个batch的输入图片的shape
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size  # final letterboxed shape
            img, ratio, pad = letterbox(img, shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            # Load labels
            labels = []
            # self.labels = [array( [对应1.txt的labels信息] ，dtype=float32), ..., array( [对应n.txt的labels信息] ，dtype=float32)]
            x = self.labels[index]  # x.size = (目标数量, [class, xywh, Θ])
            if x.size > 0:
                # Normalized xywh to pixel xyxy format
                # 根据pad调整框的标签坐标，并从归一化的xywh->未归一化的xyxy
                labels = x.copy()   # labels.size = (单张图片中的目标数量, [class, xyxy])
                labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
                labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
                labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
                labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.augment:
            # Augment imagespace
            if not mosaic:
                # 随机对图片进行旋转，平移，缩放，裁剪
                img, labels = random_perspective(img, labels,
                                                 degrees=hyp['degrees'],
                                                 translate=hyp['translate'],
                                                 scale=hyp['scale'],
                                                 shear=hyp['shear'],
                                                 perspective=hyp['perspective'])

            # Augment colorspace
            # 随机改变图片的色调（H），饱和度（S），亮度（V）
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

            # Apply cutouts
            # if random.random() < 0.9:
            #     labels = cutout(img, labels)

        # labels.size = (目标数量, [class, xyxy, Θ])
        nL = len(labels)  # number of labels
        if nL:
            # 调整框的标签，xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh

            # 重新归一化标签0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

        # labels.size = (目标数量, [class, xywh, Θ])
        if self.augment:
            # flip up-down 上下翻转  沿x轴翻转 （y变 x不变）
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]  # y变x不变
                    labels[:, -1] = 180 - labels[:, -1]  # θ根据左右偏转也进行改变
                    labels[labels[:, -1] == 180, -1] = 0  # 原θ=0时，情况特殊不做改变

            # flip left-right 左右翻转  沿y轴翻转（y不变 x变）
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]  # x变y不变
                    labels[:, -1] = 180 - labels[:, -1]  # θ根据左右偏转也进行改变
                    labels[labels[:, -1] == 180, -1] = 0  # 原θ=0时，情况特殊不做改变

            # #  旋转augment
            # if nL:
            #     degrees = 10.0
            #     rotate_angle = random.uniform(-degrees, degrees)
            #     img, labels = rotate_augment(rotate_angle, 1, img, labels)

        # 初始化标签框对应的图片序号，配合下面的collate_fn使用
        labels_out = torch.zeros((nL, 7))
        if nL:
            # labels.size=(目标数量, [class,xywh,Θ])  ->  labels_out.size=(目标数量, [?, class,xywh,Θ])
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img.size=[resized_height,resized_width,3] -> [3, resized_height, resized_width]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        '''
        img: 经预处理后的img size= [3, resized_height, resized_width]
        labels_out :  (目标数量, [0, classid,归一化后的xywh,Θ])
        self.img_files[index] : 图片索引index的文件路径
        shapes：不进行mosaic时进行矩形训练时才有值
        '''
        return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):  # 取样器取样本的函数 即可通过该函数重写并自定义聚合为batch的方式
        """
        return img, labels, path, shapes
        @param batch:  一个batch里面包含img，label，path，shapes 重写batch取样函数
        @return:
                img : size = (batch_size, 3 , resized_height, resized_width) 没有归一化
                labels : size = (batch中的目标数量, [图片在当前batch中的索引,classid,归一化后的xywh, Θ])
                        eg:[[0, 6, 0.5, 0.5, 0.26, 0.35, 179],
                            [0, 6, 0.5, 0.5, 0.26, 0.35, 179],
                            [1, 6, 0.5, 0.5, 0.26, 0.35, 179],
                            [2, 6, 0.5, 0.5, 0.26, 0.35, 179],]
                path： 该batch中所有image的路径 size=batch_size
                shapes: 该batch中所有image的shapes size=batch_size 不进行mosaic时进行矩形训练时才有值
        """
        # 一个batch中的img，标签信息，路径信息，形状信息 batch中的每个索引都由__getitem__函数提供
        # eg: label:[[1.txt的labels信息], ... ,[2.txt的labels信息]]
        img, label, path, shapes = zip(*batch)  # transposed
        for i, l in enumerate(label):  # i对应一个batch中的图片索引
            l[:, 0] = i  # add target image index for build_targets()
        # stack 和cat都是对tensor沿指定维度拼接，stack会增加一个维度，cat不会增加维度
        # img增加一个batch_size维度
        # label打破一个维度由label:[[1.txt的labels信息], ... ,[2.txt的labels信息]] -> [batch中的目标数量,[图片在当前batch中的索引,classid,归一化后的xywh]]
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


# Ancillary functions --------------------------------------------------------------------------------------------------
def load_image(self, index):
    '''
    loads 1 image from dataset 加载训练列表中的一张图片
    @param self: dataset类
    @param index: 用于索引当前训练集中的图片
    @return:
    ----------------------------------------
    若图片无缓存：
        img： 图像像素矩阵 size=(height, width, 3)
        (h0, w0)： 图像原始的（height，width）
        img.shape[:2]： 图像resize之后的（height，width）
    否则：
        self.imgs[index]： 图像像素矩阵 size=(height, width, 3)
        self.img_hw0[index]： 图像原始的（height，width）
        self.img_hw[index]： 图像resize之后的（height，width）
    ----------------------------------------
    '''
    img = self.imgs[index]
    if img is None:  # not cached
        path = self.img_files[index]
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized
    else:
        return self.imgs[index], self.img_hw0[index], self.img_hw[index]  # img, hw_original, hw_resized


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

    # Histogram equalization
    # if random.random() < 0.2:
    #     for i in range(3):
    #         img[:, :, i] = cv2.equalizeHist(img[:, :, i])


def load_mosaic(self, index):
    '''
    loads 4 images in a mosaic
    @param self:  一个dataset类
    @param index:  索引号，用于索引整个训练集合中的图片
    @return:
             ——img4 : size = (resized_height,resized_ width, 3);经
             ——labels4 : size = (单张img4中的目标GT数量, [classid ,LT_x,LT_y,RB_x,RB_y,Θ]未归一化);
    '''

    labels4 = []
    s = self.img_size
    # 随机取mosaic中心点
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]  # mosaic center x, y
    # 随机取其他三张图片的索引
    indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
    for i, index in enumerate(indices):
        # Load image
        # img.size = [resized_height,resized_ width, 3]
        img, _, (h, w) = load_image(self, index)

        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)  用于确定原图片在img4左上角的坐标（左上右下）
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)  用于确定原图片剪裁进img4中的图像内容范围
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # img4.size = [resized_height,resized_ width, 3]
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b   # 原图片未剪裁进img4中的宽度
        padh = y1a - y1b   # 原图片未剪裁进img4中的高度

        # Labels
        # self.labels[array([对应1.txt的labels信息] ，dtype = float32), ..., array([对应n.txt的labels信息] ，dtype = float32)]
        x = self.labels[index]
        labels = x.copy()
        if x.size > 0:  # Normalized xywh to pixel xyxy format 归一化的xywh转为非归一化的xyxy（左上右下）坐标形式
            labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw  # Left_top_x
            labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh  # Left_top_y
            labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw  # right_bottom_x
            labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh  # right_bottom_y
        labels4.append(labels)  # labels4：[array([对应1.txt的labels信息 size=[n1,6]], ... ,array([对应4.txt的labels信息] size=[n4,6]]

    # Concat/clip labels
    if len(labels4):
        # labels4：[array([对应1.txt的labels信息 size=[n1,6]], ... ,array([对应4.txt的labels信息] size=[n4,6]]  -> [4张图片的gt总数n1+n2+n3+n4,6]
        # 即labels4.shape=(一张mosaic图片中的GT数量, [classid ,LT_x,LT_y,RB_x,RB_y,Θ])
        labels4 = np.concatenate(labels4, 0)  # 将第一个维度取消
        np.clip(labels4[:, 1:5], 0, 2 * s, out=labels4[:, 1:5])  # 限定labels4[:, 1:5]中最小值只能为0，最大值只能为2*self.size
        # img4, labels4 = replicate(img4, labels4)  # replicate

    # Augment
    img4, labels4 = random_perspective(img4, labels4,
                                       degrees=self.hyp['degrees'],
                                       translate=self.hyp['translate'],
                                       scale=self.hyp['scale'],
                                       shear=self.hyp['shear'],
                                       perspective=self.hyp['perspective'],
                                       border=self.mosaic_border)  # border to remove

    '''
    img4 : (size1, size2, 3)
    labels4 : (单张img4中的目标GT数量, [classid ,LT_x,LT_y,RB_x,RB_y,Θ])
    '''
    return img4, labels4


def replicate(img, labels):
    # Replicate labels
    h, w = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    '''
    Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    @param new_shape:  矩形训练后的输出size
    @param color:   用于填充图片未覆盖区域的背景色
    @return:
    @param img:  待矩形训练后的输入图像
    @return:
        img ： 矩形训练后的输出图像
        ratio ： [width_ratio , height_ratio] 最终size/原始size
        (dw, dh) ：最小的左右/上下填充大小
    '''
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # 计算缩放因子
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 获取最小的矩形填充
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 计算上下左右填充大小
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 进行填充
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    '''
    遍性数据增强：
            进行随机旋转，缩放，错切，平移，center，perspective数据增强
    Args:
        img: shape=(height, width, 3)
        targets ：size = (单张图片中的目标数量, [class, xyxy, Θ])
    Returns:
        img：shape=(height, width, 3)
        targets = (目标数量, [cls, xyxy, Θ])
    '''

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # 设置旋转和缩放的仿射矩阵并进行旋转和缩放
    # Rotation and Scale
    R = np.eye(3)  # 行数为3,对角线为1,其余为0的矩阵
    a = random.uniform(-degrees, degrees)   # 随机生成[-degrees, degrees)的实数 即为旋转角度 负数则代表逆时针旋转
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)  # 获得以(0,0)为中心的旋转仿射变化矩阵

    # 设置裁剪的仿射矩阵系数
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # 设置平移的仿射系数
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 融合仿射矩阵并作用在图片上
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(img[:, :, ::-1])  # base
    # ax[1].imshow(img2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 调整框的标签
    n = len(targets)  # targets.size = (目标数量, [class, xyxy, Θ])
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # # apply angle-based reduction of bounding boxes
        # radians = a * math.pi / 180
        # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        # x = (xy[:, 2] + xy[:, 0]) / 2
        # y = (xy[:, 3] + xy[:, 1]) / 2
        # w = (xy[:, 2] - xy[:, 0]) * reduction
        # h = (xy[:, 3] - xy[:, 1]) * reduction
        # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # clip boxes
        xy_ = xy.copy()
        xy_[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy_[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy_.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

        # return unobscured labels
        if len(labels) and s > 0.03:
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def reduce_img_size(path='path/images', img_size=1024):  # from utils.datasets import *; reduce_img_size()
    # creates a new ./images_reduced folder with reduced size images of maximum size img_size
    path_new = path + '_reduced'  # reduced images path
    create_folder(path_new)
    for f in tqdm(glob.glob('%s/*.*' % path)):
        try:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            r = img_size / max(h, w)  # size ratio
            if r < 1.0:
                img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)  # _LINEAR fastest
            fnew = f.replace(path, path_new)  # .replace(Path(f).suffix, '.jpg')
            cv2.imwrite(fnew, img)
        except:
            print('WARNING: image failure %s' % f)


def recursive_dataset2bmp(dataset='path/dataset_bmp'):  # from utils.datasets import *; recursive_dataset2bmp()
    # Converts dataset to bmp (for faster training)
    formats = [x.lower() for x in img_formats] + [x.upper() for x in img_formats]
    for a, b, files in os.walk(dataset):
        for file in tqdm(files, desc=a):
            p = a + '/' + file
            s = Path(file).suffix
            if s == '.txt':  # replace text
                with open(p, 'r') as f:
                    lines = f.read()
                for f in formats:
                    lines = lines.replace(f, '.bmp')
                with open(p, 'w') as f:
                    f.write(lines)
            elif s in formats:  # replace image
                cv2.imwrite(p.replace(s, '.bmp'), cv2.imread(p))
                if s != '.bmp':
                    os.system("rm '%s'" % p)


def imagelist2folder(path='path/images.txt'):  # from utils.datasets import *; imagelist2folder()
    # Copies all the images in a text file (list of images) into a folder
    create_folder(path[:-4])
    with open(path, 'r') as f:
        for line in f.read().splitlines():
            os.system('cp "%s" %s' % (line, path[:-4]))
            print(line)


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
