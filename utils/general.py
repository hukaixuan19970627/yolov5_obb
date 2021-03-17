import glob
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm

from utils.google_utils import gsutil_getsize
from utils.torch_utils import is_parallel, init_torch_seeds
from shapely.geometry import Polygon, MultiPoint

from utils import polyiou
import pdb

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
matplotlib.rc('font', **{'size': 11})

# Prevent OpenCV from multithreading (to use PyTorch DataLoader)
cv2.setNumThreads(0)


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    Decorator使分布式训练中的所有进程等待每个本地的主进程做一些事情。
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    '''
    设置唯一确定随机数种子，确保随机数种子不变，使得程序每次使用random函数均可获得同一随机值,即确保神经网络每次初始化都相同
    '''
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='./runs'):
    '''
    Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    '''
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def check_git_status():
    '''
    Suggest 'git pull' if repo is out of date
    '''
    if platform.system() in ['Linux', 'Darwin'] and not os.path.isfile('/.dockerenv'):
        s = subprocess.check_output('if [ -d .git ]; then git fetch && git status -uno; fi', shell=True).decode('utf-8')
        if 'Your branch is behind' in s:
            print(s[s.find('Your branch is behind'):s.find('\n\n')] + '\n')


def check_img_size(img_size, s=32):
    '''
    Verify img_size is a multiple of stride s
    return new_size
    '''
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size


def check_anchors(dataset, model, thr=4.0, imgsz=640):
    '''
    Check anchor fit to data, recompute if necessary
    利用预设值anchor基于shape规则对bbox计算best possible recall
    若召回率大于一定值，则不进行优化，直接返回
    若召回率低，则利用遗传算法+kmeans重新计算anchor
    '''
    print('\nAnalyzing anchors... ', end='')
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1. / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1. / thr).float().mean()  # best possible recall
        return bpr, aat

    bpr, aat = metric(m.anchor_grid.clone().cpu().view(-1, 2))
    print('anchors/target = %.2f, Best Possible Recall (BPR) = %.4f' % (aat, bpr), end='')
    if bpr < 0.98:  # threshold to recompute
        print('. Attempting to generate improved anchors, please wait...' % bpr)
        na = m.anchor_grid.numel() // 2  # number of anchors
        new_anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(new_anchors.reshape(-1, 2))[0]
        if new_bpr > bpr:  # replace anchors
            new_anchors = torch.tensor(new_anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchor_grid[:] = new_anchors.clone().view_as(m.anchor_grid)  # for inference
            m.anchors[:] = new_anchors.clone().view_as(m.anchors) / m.stride.to(m.anchors.device).view(-1, 1, 1)  # loss
            check_anchor_order(m)
            print('New anchors saved to model. Update model *.yaml to use these anchors in the future.')
        else:
            print('Original anchors better than new anchors. Proceeding with original anchors.')
    print('')  # newline


def check_anchor_order(m):
    '''
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    # 检查YOLOv5 Detect（）模块m的anchor顺序和stride顺序，如有必要，进行纠正，确保anchor顺序是从小物体的anchor到大物体的anchor
    @param m: Detect类
    '''
    # prod返回指定数轴上所有元素的乘积  view(-1)将数据展开为一维数组
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def check_file(file):
    '''
    Search for file if not found
    '''
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def check_dataset(dict):
    '''
    Download dataset if not found
    '''
    val, s = dict.get('val'), dict.get('download')
    if val and len(val):
        val = [os.path.abspath(x) for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(os.path.exists(x) for x in val):
            print('\nWARNING: Dataset not found, nonexistant paths: %s' % [*val])
            if s and len(s):  # download script
                print('Downloading %s ...' % s)
                if s.startswith('http') and s.endswith('.zip'):  # URL
                    f = Path(s).name  # filename
                    torch.hub.download_url_to_file(s, f)
                    r = os.system('unzip -q %s -d ../ && rm %s' % (f, f))  # unzip
                else:  # bash script
                    r = os.system(s)
                print('Dataset autodownload %s\n' % ('success' if r == 0 else 'failure'))  # analyze return value
            else:
                raise Exception('Dataset not found.')


def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor , 返回可被除数divisor整除的x,否则返回divisor
    return math.ceil(x / divisor) * divisor


def labels_to_class_weights(labels, nc=80):
    '''
    Get class weights (inverse frequency) from training labels 获取图像的采样权重（图像类别的反频率：图像类别频率高的采样频率低）
    '''
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]  classes : size=(866643)
    weights = np.bincount(classes, minlength=nc)  # occurences per class 输出长度为nc的数组，其中数值为每一类别出现的频数

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1  种类频数为0，则用1来填充
    weights = 1 / weights  # number of targets per class  频数取反，频数越高，反而此时的数值越低（频率取反）
    weights /= weights.sum()  # normalize  求出每个类别的占总数的反比例
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array([np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    '''
    Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    '''
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def scale_labels(img1_shape, labels, img0_shape, ratio_pad=None):
    '''
    Rescale coords (xywh) from img1_shape to img0_shape
    将检测出的目标边框坐标从 img1_shape 形状放缩到 img0_shape，即反resize+pad，将目标边框对应至初始原图
    @param img1_shape:  原始形状 (height, width)
    @param labels: (num ,[ x y longside shortside Θ])
    @param img0_shape:  目标形状 (height, width)
    @param ratio_pad:
    @return:
            scaled_labels : (num ,[ x y longside shortside Θ])
    '''
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    scaled_labels = []
    for i, label in enumerate(labels):
        # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
        rect = longsideformat2cvminAreaRect(label[0], label[1], label[2], label[3], (label[4] - 179.9))
        # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        poly = cv2.boxPoints(rect)  # 返回rect对应的四个点的值 normalized

        poly[:, 0] -= pad[0]   # x padding
        poly[:, 1] -= pad[1]   # y padding
        poly[:, :] /= gain
        clip_poly(poly, img0_shape)

        rect_scale = cv2.minAreaRect(np.float32(poly))  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）

        c_x = rect_scale[0][0]
        c_y = rect_scale[0][1]
        w = rect_scale[1][0]
        h = rect_scale[1][1]
        theta = rect_scale[-1]  # Range for angle is [-90，0)

        label = np.array(cvminAreaRect2longsideformat(c_x, c_y, w, h, theta))

        label[-1] = int(label[-1] + 180.5)  # range int[0,180] 四舍五入
        if label[-1] == 180:  # range int[0,179]
            label[-1] = 179
        scaled_labels.append(label)

    return torch.from_numpy(np.array(scaled_labels))


def clip_poly(poly, img_shape):
    '''
    Clip bounding [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] bounding boxes to image shape (height, width)
    '''
    poly[:, 0].clip(0, img_shape[1])  # x
    poly[:, 1].clip(0, img_shape[0])  # y

def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, fname='precision-recall_curve.png'):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        fname:  Plot filename
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 or n_gt == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_gt + 1e-16)  # recall curve
            r[ci] = np.interp(-pr_score, -conf[i], recall[:, 0])  # r at pr_score, negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            py.append(np.interp(px, recall[:, 0], precision[:, 0]))  # precision at mAP@0.5
            for j in range(tp.shape[1]):
                ap[ci, j] = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    if plot:
        py = np.stack(py, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(px, py, linewidth=0.5, color='grey')  # plot(recall, precision)
        ax.plot(px, py.mean(1), linewidth=2, color='blue', label='all classes')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.legend()
        fig.tight_layout()
        fig.savefig(fname, dpi=200)

    return p, r, ap, f1, unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.], recall, [min(recall[-1] + 1E-3, 1.)]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

# 计算旋转矩形iou
def rotate_box_iou(box1, box2, GIoU=False):
    """
    计算box1中的所有box 与 box2中的所有box的旋转矩形iou （1对1）
    :param box1: GT tensor  size=(n, [xywhθ])
    :param box2: anchor  size= (n, [xywhθ])
    :param GIoU: 是否使用GIoU的标志位
    :return:
             box2所有box与box1的IoU  size= (n)
    """
    ft = torch.cuda.FloatTensor
    if isinstance(box1, list):  box1 = ft(box1)
    if isinstance(box2, list):  box2 = ft(box2)

    if len(box1.shape) < len(box2.shape):  # 输入的单box维度不匹配时，unsqueeze一下 确保两个维度对应两个维度
        box1 = box1.unsqueeze(0)
    if len(box2.shape) < len(box1.shape):  # 输入的单box维度不匹配时，unsqueeze一下 确保两个维度对应两个维度
        box2 = box2.unsqueeze(0)
    if not box1.shape == box2.shape:  # 若两者num数量不等则报错
        print('计算旋转矩形iou时有误，输入shape不相等')
        print('----------------box1:--------------------')
        print(box1.shape)
        print(box1)
        print('----------------box2:--------------------')
        print(box2.shape)
        print(box2)
    # print(box1)
    # box(n, [xywhθ])
    box1 = box1[:, :5]
    box2 = box2[:, :5]

    if GIoU:
        mode = 'giou'
    else:
        mode = 'iou'

    ious = []
    for i in range(len(box2)):
        # print(i)
        r_b1 = get_rotated_coors(box1[i])
        r_b2 = get_rotated_coors(box2[i])

        ious.append(skewiou(r_b1, r_b2, mode=mode))

    # if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
    #     c_x1, c_x2 = torch.min(b1_x1, b2_x1), torch.max(b1_x2, b2_x2)
    #     c_y1, c_y2 = torch.min(b1_y1, b2_y1), torch.max(b1_y2, b2_y2)
    #     c_area = (c_x2 - c_x1) * (c_y2 - c_y1)  # convex area
    #     return iou - (c_area - union_area) / c_area  # GIoU
    # print(ious)
    return ft(ious)

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    """
        Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    @param box1: shape([xy_offsets_feature,wh_feature], num)
    @param box2: shape(num, [xy_offsets_feature,wh_feature])
    @param x1y1x2y2: bbox的表示形式是否已经是xyxy？
    @return:  iou   shape=(num, 1)
    """
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps  # eps防止分母变为0
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union   # IoU= (A∩B)/(A∪B)
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width 并集外接矩形的宽
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height 并集外接矩形的高
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared  并集最小外接矩形的对角线距离(squared)
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared 两矩形中心点欧氏距离的平方
            if DIoU:
                return iou - rho2 / c2  # DIoU = IoU - 惩罚项
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU = DIou - αv
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area 并集外接矩形的面积
            return iou - (c_area - union) / c_area  # GIoU = IoU - (C-A∪B)/C
    else:
        return iou  # IoU


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = torch.min(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    '''
    return positive, negative label smoothing BCE targets
    '''
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()

def gaussian_label(label, num_class, u=0, sig=4.0):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    return np.concatenate([y_sig[math.ceil(num_class / 2) - int(label.item()):],
                           y_sig[:math.ceil(num_class / 2) - int(label.item())]], axis=0)

def rbox_iou(box1, theta1, box2, theta2):
    """
    compute rotated box IoU
    @param box1: torch.size(num, 4)
    @param theta1: torch.size(num, 1)
    @param box2: torch.size(num, 4)
    @param theta2: torch.size(num, 1)
    @return:
             rbox_iou  numpy_array shape(num, 1)
    """
    # theta2 = theta2.unsqueeze(1)  # torch.size  num -> (num,1)
    polys1 = []
    polys2 = []
    rboxes1 = torch.cat((box1, theta1), 1)
    rboxes2 = torch.cat((box2, theta2), 1)
    for rbox1 in rboxes1:
        poly = longsideformat2poly(rbox1[0], rbox1[1], rbox1[2], rbox1[3], rbox1[4])
        polys1.append(polyiou.VectorDouble(poly))
    for rbox2 in rboxes2:
        poly = longsideformat2poly(rbox2[0], rbox2[1], rbox2[2], rbox2[3], rbox2[4])
        polys2.append(polyiou.VectorDouble(poly))
    IoUs = []
    for i in range(len(polys1)):
        iou = polyiou.iou_poly(polys1[i], polys2[i])
        IoUs.append(iou)
    IoUs = np.array(IoUs)

    return IoUs



def compute_loss(p, targets, model, csl_label_flag=True):
    '''
    @param p: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, no)
    @param targets: torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh,Θ])
    @param model: 网络模型
    @param csl_label_flag: θ是否采用CSL_labels来计算分类损失
    @return:
            loss * bs : 标量  ；
            torch.cat((lbox, lobj, lcls, langle, loss)).detach() : 不参与网络更新的标量 list(边框损失, 置信度损失, 分类损失, 角度loss,总损失)
    '''
    device = targets.device
    # 初始化各个部分损失
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    langle = torch.zeros(1, device=device)
    # 获得标签分类，边框，索引，anchor
    '''
        tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor
                       tcls[i].shape=(num_i, 1)
        tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh
                       tbox[i].shape=(num_i, 4)
        indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
                       (该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
                             indices[i].shape=(4, num_i)
        anchors : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
                            anchor[i].shape=(num_i, 2)
        tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor
                       tangle[i].shape=(num_i)
    '''
    tcls, tbox, indices, anchors, tangle = build_targets(p, targets, model)  # targets
    h = model.hyp  # hyperparameters

    # Define criteria
    # 定义损失函数 分类损失和 置信度损失
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['cls_pw']])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['obj_pw']])).to(device)
    BCEangle = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h['angle_pw']])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    # 标签平滑，eps默认为0，其实是没用上 cp = 1; cn = 0
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    # 如果设置了fl_gamma参数，就使用focal loss，默认也是没使用的
    g = h['fl_gamma']  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        BCEangle = FocalLoss(BCEangle, g)

    # Losses

    nt = 0  # number of targets
    np = len(p)  # number of inference outputs = 3
    # 设置三个特征图对应输出的损失系数  4.0, 1.0, 0.4分别对应下采样8,16,32的输出层
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        # 根据indices获取索引，方便找到对应网格的输出
        # pi.size = (batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angles])
        # indice[i] = (该image属于该batch的第几个图片 ,该box属于哪种scale的anchor，网格索引1，网格索引2)
        # indices[i].shape=(4, num_i)
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx  shape=(num_i)
        # tobj.size = (batch_size, 3种scale框, feature_height, feature_width, 1) 全为0
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of GT_targets_filter  num
        if n:
            nt += n  # cumulative targets 累加三个检测层中的gt数量
            # 前向传播结果pi.shape(batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angles])
            # b, a, gj, gi  shape均=(num_filter)经过筛选的gt信息 pi[该image属于该batch的第几个图片，该box属于哪种scale的anchor，网格索引1，网格索引2]
            # 得到ps.size = (经过与gt匹配后筛选的数量N ,[xywh,score,num_classes,num_angles])
            import numpy
            # print(pi.shape)
            # numpy.savetxt("b.txt", b.numpy(), fmt='%f', delimiter=',')
            # print(b.numpy())
            # numpy.savetxt("a.txt", a.numpy(), fmt='%f', delimiter=',')
            # print(a.numpy())
            # numpy.savetxt("gj.txt", gj.numpy(), fmt='%f', delimiter=',')
            # print(gj.numpy())
            # numpy.savetxt("gi.txt", gi.numpy(), fmt='%f', delimiter=',')
            # print(gi.numpy())
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets  前向传播结果与target信息进行匹配 筛选对应的网格 得到对应网格的前向传播结果

            # Regression
            # pxy.shape(num, 2)
            pxy = ps[:, :2].sigmoid() * 2. - 0.5  # 对前向传播结果xy进行回归  （预测的是offset）-> 处理成与当前网格左上角坐标的xy偏移量
            # pxy.shape(num, 2)
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]  # 对前向传播结果wh进行回归  （预测的是当前featuremap尺度上的框wh尺度缩放量）-> 处理成featuremap尺度上框的真实wh
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box 生成featuremap上的bbox  shape(num, 4)
            # 计算边框损失，注意这个CIoU=True，计算的是ciou损失
            # 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box tensor
            # pbox.T.shape=(4, num)        tbox[i].shape=(num, 4)
            iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)  shape=(num)
            lbox += (1.0 - iou).mean()  # iou loss  iou为两者的匹配度 因此计算loss时必须让匹配度高的loss贡献更低  因此1-iou后取num个数据的均值  shape(1)

            # Classification  设置如果类别数大于1才计算分类损失
            class_index = 5 + model.nc
            if model.nc > 1:  # cls loss (only if multiple classes)
                # ps.size = (经过与gt匹配后筛选的数量N ,[xywh,score,num_classes,num_angles])
                # t.size = (num ,num_classes) 值全为cn=0（没做标签平滑）
                t = torch.full_like(ps[:, 5:class_index], cn, device=device)  # targets
                # tcls[i] : 对当前步长网络生成对应的class tensor  tcls[i].shape=(num, 1)  eg：tcls[0] = tensor([73, 73, 73])
                # 在num_classes处对应的类别位置置为cp=1 （没做标签平滑）  i为layer index
                t[range(n), tcls[i]] = cp

                # 前向传播结果与targets结果开始计算分类损失并累加
                # 筛选后的前向传播结果ps[:, 5:].shape=(num, num_classes)   t.shape=(num ,num_classes)
                lcls += BCEcls(ps[:, 5:class_index], t)  # BCE 分类损失以BCEWithLogitsLoss来计算

            # Θ类别损失
            if not csl_label_flag:
                ttheta = torch.full_like(ps[:, class_index:], cn, device=device)
                ttheta[range(n), tangle[i]] = cp
                langle += BCEangle(ps[:, class_index:], ttheta)  # BCE Θ类别损失以BCEWithLogitsLoss来计算
            else:
                ttheta = torch.zeros_like(ps[:, class_index:])  # size(num, 180)
                for idx in range(len(ps)):  # idx start from 0 to len(ps)-1
                    # 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor  tangle[i].shape=(num_i, 1)
                    theta = tangle[i][idx]  # 取出第i个layer中的第idx个目标的角度数值  例如取值θ=90
                    # CSL论文中窗口半径为6效果最佳，过小无法学到角度信息，过大则角度预测偏差加大
                    csl_label = gaussian_label(theta, 180, u=0, sig=6)  # 用长度为1的θ值构建长度为180的csl_label
                    ttheta[idx] = torch.from_numpy(csl_label)  # 将cls_label放入对应的目标中
                langle += BCEangle(ps[:, class_index:], ttheta)

            angle_ = ps[:, class_index:]
            angle_value_, angle_index_ = torch.max(angle_, 1, keepdim=True)  # size都为 (num, 1)
            riou = torch.from_numpy(rbox_iou(pbox, angle_index_, tbox[i], tangle[i].unsqueeze(1))).cuda()
            # Objectness 置信度
            # 根据model.gr设置objectness的标签值
            # tobj.size = (batch_size, 3种scale框, size1, size2, 1) 表示该网格预测的是前景（1）还是背景（0）
            # 使用标签框与预测框的CIoU值来作为该预测框的conf分支的权重系数 detach不参与网络更新  (1.0 - model.gr)为objectness额外的偏移系数
            tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * riou.detach().clamp(0).type(
                tobj.dtype)  # iou ratio 与target信息进行匹配 筛选为前景的网格 shape(num)

        # 计算objectness的损失  计算score与labels的损失
        # pi.size = (batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angles])
        # tobj.size = (batch_size, 3种scale框, size1, size2, 1) 其中与gt对应的位置为当前预测框与gt框的?IoU值 ；预测框与gt框的匹配度越高理应预测质量越高
        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss 最后分别乘上3个尺度检测层的权重并累加

    # 根据超参数设置的各个部分损失的系数 获取最终损失
    s = 3 / np  # output count scaling
    lbox *= h['box'] * s
    lobj *= h['obj'] * s * (1.4 if np == 4 else 1.)
    lcls *= h['cls'] * s
    langle *= h['angle'] * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls + langle
    '''
    loss * bs : 标量
    torch.cat((lbox, lobj, lcls, langle, loss)) : 不参与网络更新的标量 list(边框损失, 置信度损失, 分类损失, Θ分类损失,总损失)
    '''
    return loss * bs, torch.cat((lbox, lobj, lcls, langle, loss)).detach()


def build_targets(p, targets, model):
    """
        Build targets for compute_loss(), input targets(image,class,x,y,w,h)；
    Args :
        predictions :[small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, no)
        targets : torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, Θ])
        model : 模型

    Returns:
        tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor
                       tcls[i].shape=(num_i, 1)
                   eg：tcls[0] = tensor([73, 73, 73])
        tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh
                       tbox[i].shape=(num_i, 4)
                   eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                         [ 1.19355,  0.27958,  4.38709, 14.92512],
                                         [ 0.19355,  1.27958,  4.38709, 14.92512]])
        indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
                       (该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
                             indices[i].shape=(4, num_i)
                        eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
        anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
                            anchor[i].shape=(num_i, 2)
                        eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
        tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的angle tensor
                       tangle[i].shape=(num_i, 1)
                   eg：tangle[0] = tensor([179, 179, 179])
    """
    # 获取每一个(3个)检测层
    det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
    # anchor数量和GT标签框数量
    na, nt = det.na, targets.shape[0]  # number of anchors=3, nums of targets in one batch
    tcls, tbox, indices, anch = [], [], [], []
    tangle = []
    gain = torch.ones(8, device=targets.device)  # normalized to gridspace gain
    # ai.shape = (3, nt) 生成anchor索引  anchor index; ai[0]全等于0. ai[1]全等于1. ai[2]全等于2.用于表示当前gtbox和当前层哪个anchor匹配
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    '''
    targets.size(该batch中的GT数量, 7)  ->   targets.size(3(原来数据的基础上重复三次，按行拼接), 该batch中的GT数量, 7)
    targets.size(3(原来数据的基础上重复三次，按行拼接), 该batch中的GT数量, 7) ->  targets.size(3(原来数据的基础上重复三次，按行拼接), 该batch中的GT数量, 7 + anchor_index)
    targets.shape = (3, num_gt_batch, [该image属于该batch的第几个图片, class, xywh,Θ, 用第几个anchor进行检测])
    由于每个尺度的feature map各自对应3种scale的anchor，因此将GT标签信息重复三次，方便与每个点的3个anchor单独匹配
    '''
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    # 设置偏移量
    g = 0.5  # bias 网格中心偏移
    # 附近的四个网格 off.shape = (5, 2)
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    # 对每个检测层进行处理
    for i in range(det.nl):  # 3种步长的feature map
        # det.anchor(3, 3, 2)  anchors: -> 原始anchor(0,:,:)/ 8. , anchor(1,:,:)/ 16.  anchor(2,:,:)/ 32.
        # anchors (3, 2)  3种scale的anchor wh
        anchors = det.anchors[i]  # small->medium->large anchor框
        # 得到特征图的坐标系数
        """
        p[i].shape = (b, 3种scale框, h, w, [xywh,score,num_classes,num_angle]), hw分别为特征图的长宽
        gain = [1, 1, h, w, h, w, 1, 1]
        """
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain  把p[i]wh维度的数据赋给gain
        num_h, num_w = p[i].shape[2:4]

        # Match targets to anchors
        # targets.shape = (3, num_gt_batch, [该image属于该batch的第几个图片, class, xywh,Θ, 用哪个anchor进行检测])  gain = [1, 1, w, h, w, h, 1]
        # t.shape = (3 , num_gt_batch, [该image属于该batch的第几个图片, class, xywh_feature,Θ, 用哪个anchor进行检测])
        t = targets * gain  # 将labels的归一化的xywh从基于0~1映射到基于特征图的xywh 即变成featuremap尺度

        if nt:  # num_targets 该batch中的目标数量
            # Matches
            """
            GT的wh与anchor的wh做匹配，筛选掉比值大于hyp['anchor_t']的(这应该是yolov5的创新点)targets，从而更好的回归(与新的边框回归方式有关)
            若gt_wh/anhor_wh 或 anhor_wh太大/gt_wh 超出hyp['anchor_t']，则说明当前target与所选anchor形状匹配度不高，该物体宽高过于极端，不应强制回归，将该处的labels信息删除，在该层预测中认为是背景
            
            由于yolov3回归wh采用的是out=exp(in)，这很危险，因为out=exp(in)可能会无穷大，就会导致失控的梯度，不稳定，NaN损失并最终完全失去训练；
            (当然原yolov3采用的是将targets进行反算来求in与网络输出的结果，就问题不大，但采用iou loss，就需要将网络输出算成out来进行loss求解，所以会面临这个问题)；
            所以作者采用新的wh回归方式:
            (wh.sigmoid() * 2) ** 2 * anchors[i], 原来yolov3为anchors[i] * exp(wh)
            将标签框与anchor的倍数控制在0~4之间；
            hyp.scratch.yaml中的超参数anchor_t=4，所以也是通过此参数来判定anchors与标签框契合度；
            """
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio 获取gt bbox与anchor的wh比值  shape=(3, num_gt_batch, 2)
            j = torch.max(r, 1. / r).max(2)[0] < model.hyp['anchor_t']  # compare   shape=(3,num_gt_batch)
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            '''
            从(3 , num_gt_targets_thisbatch,8) 的t中筛选符合条件的anchor_target框
            即 3 * num_gt_targets_thisbatch个anchor中筛出M个有效GT
            经过第i层检测层筛选过后的t.shape = (M, 8),M为筛选过后的数量
            '''
            # shape=(3 , num_gt_batch_filter, 8) ->  (M, [该image属于该batch的第几个图片, class, xywh_feature,Θ, 用哪个anchor进行检测])
            t = t[j]  # filter 筛选出与anchor匹配的targets;

            # Offsets
            # 得到筛选后的GT的中心点坐标xy-featuremap尺度(相对于左上角的), 其shape为(M, [x_featuremap, y_featuremap])
            gxy = t[:, 2:4]  # grid gt xy
            # 得到筛选后的GT的中心点相对于右下角的坐标, 其shape为(M, 2)
            # gain = [1, 1, w, h, w, h, 1, 1]
            gxi = gain[[2, 3]] - gxy  # inverse grid gt xy
            """
            把相对于各个网格左上角x<g=0.5,y<0.5和相对于右下角的x<0.5,y<0.5的框提取出来；
            也就是j,k,l,m，在选取gij(也就是标签框分配给的网格的时候)对这四个部分的框都做一个偏移(减去上面的off),也就是下面的gij = (gxy - offsets).long()操作；
            再将这四个部分的框与原始的gxy拼接在一起，总共就是五个部分；
            也就是说：①将每个网格按照2x2分成四个部分，每个部分的框不仅采用当前网格的anchor进行回归，也采用该部分相邻的两个网格的anchor进行回归；
            原yolov3就仅仅采用当前网格的anchor进行回归；
            估计是用来缓解网格效应，但由于v5没发论文，所以也只是推测，yolov4也有相关解决网格效应的措施，是通过对sigmoid输出乘以一个大于1的系数；
            这也与yolov5新的边框回归公式相关；
            由于①，所以中心点回归也从yolov3的0~1的范围变成-0.5~1.5的范围；
            所以中心点回归的公式变为：
            xy.sigmoid() * 2. - 0.5 + cx
            """
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T  # 判断筛选后的GT中心坐标是否相对于各个网格的左上角偏移<0.5 同时 判断 是否不处于最左上角的网格中 （xy两个维度）
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T  # 判断筛选后的GT中心坐标是否相对于各个网格的右下角偏移<0.5 同时 判断 是否不处于最右下角的网格中 （xy两个维度）
            j = torch.stack((torch.ones_like(j), j, k, l, m))  # shape(5, M) 其中元素为True或False
            # 由于预设的off为5 先将t在第一个维度重复5次 shape(5, M, 8),现在选出最近的3个(包括 0，0 自己)
            t = t.repeat((5, 1, 1))[j]  # 得到经过第二次筛选的框(3*M, 8)

            # 添加偏移量  gxy.shape=(M, 2) off.shape = (5, 2)  ->  shape(5, M, 2)
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]  # 选出最近的三个网格 offsets.shape=(3*M, 2)

        else:
            t = targets[0]
            offsets = 0

        # Define
        # t.size = (3*M, [该image属于该batch的第几个图片, class, xywh_feature,Θ, 用哪个anchor进行检测])
        # b为batch中哪一张图片的索引，c为类别,angle = Θ
        b, c = t[:, :2].long().T  # image, class
        angle = t[:, 6].long()
        gxy = t[:, 2:4]  # grid xy  不考虑offset时负责预测的网格坐标 xy_featuremap 即feature尺度上的gt真实xy
        gwh = t[:, 4:6]  # grid wh  wh_featuremap
        gij = (gxy - offsets).long()  # featuremap上的gt真实xy坐标减去偏移量再取整  即计算当前label落在哪个网格坐标上
        gi, gj = gij.T  # grid xy indices 将x轴坐标信息压入gi 将y轴坐标索引信息压入gj 负责预测网格具体的整数坐标 比如 23， 2
        gi = torch.clamp(gi, 0, num_w-1)  # 确保网格索引不会超过数组的限制
        gj = torch.clamp(gj, 0, num_h-1)

        # Append
        a = t[:, 7].long()  # anchor indices  表示用第几个anchor进行检测 shape(3*M, 1)
        indices.append((b, a, gj, gi))  # image_index, anchor_index, grid indices ; 每个预测层的shape(4, 3*M)
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # 每个预测层的box shape(3*M, 4)  其中xy:当前featuremap尺度上的真实gt xy与负责预测网格坐标的偏移量  wh：当前featuremap尺度上的真实gt wh
        anch.append(anchors[a])  # anchors  每个预测层的shape(3*M, 2) 当前featuremap尺度上的anchor wh
        tcls.append(c)  # class  每个预测层的shape(3*M, 1)
        tangle.append(angle)  # angle 每个预测层的shape(3*M, 1)
    '''
    tcls : 3个tensor组成的list (tensor_class_list[i])  对每个步长网络生成对应的class tensor  
                   tcls[i].shape=(num_i, 1)  
               eg：tcls[0] = tensor([73, 73, 73])  
    tbox : 3个tensor组成的list (box[i])  对每个步长网络生成对应的gt_box信息 xy：当前featuremap尺度上的真实gt_xy与负责预测网格坐标的偏移量; wh：当前featuremap尺度上的真实gt_wh
                   tbox[i].shape=(num_i, 4)  
               eg: tbox[0] = tensor([[ 0.19355,  0.27958,  4.38709, 14.92512],
                                     [ 1.19355,  0.27958,  4.38709, 14.92512],
                                     [ 0.19355,  1.27958,  4.38709, 14.92512]])
    indices : 索引列表 也由3个大list组成 每个list代表对每个步长网络生成的索引数据。其中单个list中的索引数据分别有:
                          (该image属于该batch的第几个图片 ; 该box属于哪种scale的anchor; 网格索引1; 网格索引2)
                          indices[i].shape=(4, num_i)
                          eg： indices[0] = (tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([23, 23, 22]), tensor([2, 1, 2]))
    anch : anchor列表 也由3个list组成 每个list代表每个步长网络对gt目标采用的anchor大小(对应featuremap尺度上的anchor_wh)
                         anchor[i].shape=(num_i, 2)
                         eg：tensor([[2.00000, 3.75000],  [2.00000, 3.75000],  [2.00000, 3.75000]])
    tangle : 3个tensor组成的list (tensor_angle_list[i])  对每个步长网络生成对应的class tensor
                       tangle[i].shape=(num_i, 1)
                   eg：tangle[0] = tensor([179, 179, 179])
    '''
    return tcls, tbox, indices, anch, tangle


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    '''
    Performs Non-Maximum Suppression (NMS) on inference results；
    @param prediction:  size=(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    @param conf_thres:
    @param iou_thres:
    @param merge:
    @param classes:
    @param agnostic:
    @return:
            detections with shape: (batch_size, num_nms_boxes, [])
    '''

    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,Θ])
    nc = prediction[0].shape[1] - 5  # number of classes
    class_index = nc + 5
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image 单帧图片中的最大目标数量
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes,[xywh,score,num_classes,Θ])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, [xywh,score,num_classes,Θ])

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        # angle_index为预测的θ类别  θ ∈ int[0,179]
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
        # box.size = (num_confthres_boxes, [xywhθ])  θ ∈ [-pi/2, pi/2) length=180
        box = torch.cat((x[:, :4], (angle_index - 90) * np.pi / 180), 1)


        # Detections matrix nx7 (xywhθ, conf, clsid) θ ∈ [-pi/2, pi/2)
        if multi_label:
            # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
            # x：(num_confthres_boxes, [xywh,score,num_classes,num_angle])
            # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
            # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)  # None即新增一个维度 让每个数值单独成为一个维度

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class 按类别筛选
        if classes:
            # list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]  # any（1）函数表示每行满足条件的返回布尔值

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        # x : (num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ ∈ [-pi/2, pi/2)
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classesid*4096
        boxes, scores = x[:, :5], x[:, 5]  # boxes[x, y, w, h, θ] θ is 弧度制[-pi/2, pi/2)
        boxes[:, :4] = boxes[:, :4] + c  # boxes xywh(offset by class)



        if i.shape[0] > max_det:  # limit detections  限制单帧图片中检测出的目标数量
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        # output: (batch_size, num_nms_boxes, [x_LT,y_LT,x_RB,y_RB]+conf+class)
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def get_rotated_coors(box):
    """
    return box coors
    @param box:
    @return:
    """
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)  # 获得仿射变化矩阵
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2]
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2]
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2]
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2]
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2]
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2]
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2]
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2]

    if isinstance(x0,torch.Tensor):
        r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
                         x1.unsqueeze(0),y1.unsqueeze(0),
                         x2.unsqueeze(0),y2.unsqueeze(0),
                         x3.unsqueeze(0),y3.unsqueeze(0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box

# anchor对齐阶段计算iou
def skewiou(box1, box2,mode='iou',return_coor = False):
    a=box1.reshape(4, 2)
    b=box2.reshape(4, 2)
    # 所有点的最小凸的表示形式，四边形对象，会自动计算四个点，最后顺序为：左上 左下  右下 右上 左上
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    if not poly1.is_valid or not poly2.is_valid:
        print('formatting errors for boxes!!!! ')
        return 0
    if poly1.area == 0 or poly2.area == 0:
        return 0

    inter = Polygon(poly1).intersection(Polygon(poly2)).area
    if   mode == 'iou':
        union = poly1.area + poly2.area - inter
    elif mode =='tiou':
        union_poly = np.concatenate((a,b))   #合并两个box坐标，变为8*2
        union = MultiPoint(union_poly).convex_hull.area
        coors = MultiPoint(union_poly).convex_hull.wkt
    elif mode == 'giou':
        union_poly = np.concatenate((a,b))
        union = MultiPoint(union_poly).envelope.area
        coors = MultiPoint(union_poly).envelope.wkt
    elif mode == 'r_giou':
        union_poly = np.concatenate((a,b))
        union = MultiPoint(union_poly).minimum_rotated_rectangle.area
        coors = MultiPoint(union_poly).minimum_rotated_rectangle.wkt
    else:
        print('incorrect mode!')

    if union == 0:
        return 0
    else:
        if return_coor:
            return inter/union,coors
        else:
            return inter/union



def py_cpu_nms_poly(dets, scores,thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引  list
    """
    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)

    # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素
    order = scores.argsort()[::-1]  # 取出元素的索引值 顺序为从大到小
    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        for j in range(order.size - 1):  # 求出置信度最大poly与其他所有poly的IoU
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        inds = np.where(ovr <= thresh)[0]  # 找出iou小于阈值的索引
        order = order[inds + 1]
    return keep

def rotate_non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False, without_iouthres=False):
    """
    Performs Rotate-Non-Maximum Suppression (RNMS) on inference results；
    @param prediction: size=(batch_size, num, [xywh,score,num_classes,num_angles])
    @param conf_thres: 置信度阈值
    @param iou_thres:  IoU阈值
    @param merge: None
    @param classes: None
    @param agnostic: 进行nms是否将所有类别框一视同仁，默认False
    @param without_iouthres : 本次nms不做iou_thres的标志位  默认为False
    @return:
            output：经nms后的旋转框(batch_size, num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
    """
    # prediction :(batch_size, num_boxes, [xywh,score,num_classes,num_angles])
    nc = prediction[0].shape[1] - 5 - 180  # number of classes = no - 5 -180
    class_index = nc + 5
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections 要求冗余检测
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    # output: (batch_size, ?)
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x ： (num_boxes, [xywh, score, num_classes, num_angles])
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, no)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
        # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
        angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
        # box.size = (num_confthres_boxes, [xywhθ])  θ∈[0,179]
        box = torch.cat((x[:, :4], angle_index), 1)
        if multi_label:
            # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
            # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
            # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
            i, j = (x[:, 5:class_index] > conf_thres).nonzero(as_tuple=False).T
            # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ∈[0,179]
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:  # best class only
            conf, j = x[:, 5:class_index].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if without_iouthres:  # 不做nms_iou
            output[xi] = x
            continue

        # Filter by class 按类别筛选
        if classes:
            # list x：(num_confthres_boxes, [xywhθ,conf,classid])
            x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)] # any（1）函数表示每行满足条件的返回布尔值

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]
        # Batched NMS
        # x：(num_confthres_boxes, [xywhθ,conf,classid]) θ∈[0,179]
        c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        # boxes:(num_confthres_boxes, [xy])  scores:(num_confthres_boxes, 1)
        # agnostic用于 不同类别的框仅跟自己类别的目标进行nms   (offset by class) 类别id越大,offset越大
        boxes_xy, box_whthetas, scores = x[:, :2] + c, x[:, 2:5], x[:, 5]
        rects = []
        for i, box_xy in enumerate(boxes_xy):
            rect = longsideformat2poly(box_xy[0], box_xy[1], box_whthetas[i][0], box_whthetas[i][1], box_whthetas[i][2])
            rects.append(rect)
        i = np.array(py_cpu_nms_poly(np.array(rects), np.array(scores.cpu()), iou_thres))
        #i = nms(boxes, scores)  # i为数组，里面存放着boxes中经nms后的索引

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]

        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def strip_optimizer(f='weights/best.pt', s=''):  # from utils.general import *; strip_optimizer()
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    x['optimizer'] = None
    x['training_results'] = None
    x['epoch'] = -1
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print('Optimizer stripped from %s,%s %.1fMB' % (f, (' saved as %s,' % s) if s else '', mb))


def coco_class_count(path='../coco/labels/train2014/'):
    # Histogram of occurrences per class
    nc = 80  # number classes
    x = np.zeros(nc, dtype='int32')
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        x += np.bincount(labels[:, 0].astype('int32'), minlength=nc)
        print(i, len(files))


def coco_only_people(path='../coco/labels/train2017/'):  # from utils.general import *; coco_only_people()
    # Find images with only people
    files = sorted(glob.glob('%s/*.*' % path))
    for i, file in enumerate(files):
        labels = np.loadtxt(file, dtype=np.float32).reshape(-1, 5)
        if all(labels[:, 0] == 0):
            print(labels.shape[0], file)


def crop_images_random(path='../images/', scale=0.50):  # from utils.general import *; crop_images_random()
    # crops images into random squares up to scale fraction
    # WARNING: overwrites images!
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        img = cv2.imread(file)  # BGR
        if img is not None:
            h, w = img.shape[:2]

            # create random mask
            a = 30  # minimum size (pixels)
            mask_h = random.randint(a, int(max(a, h * scale)))  # mask height
            mask_w = mask_h  # mask width

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            cv2.imwrite(file, img[ymin:ymax, xmin:xmax])


def coco_single_class_labels(path='../coco/labels/train2014/', label_class=43):
    # Makes single-class coco datasets. from utils.general import *; coco_single_class_labels()
    if os.path.exists('new/'):
        shutil.rmtree('new/')  # delete output folder
    os.makedirs('new/')  # make new output folder
    os.makedirs('new/labels/')
    os.makedirs('new/images/')
    for file in tqdm(sorted(glob.glob('%s/*.*' % path))):
        with open(file, 'r') as f:
            labels = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
        i = labels[:, 0] == label_class
        if any(i):
            img_file = file.replace('labels', 'images').replace('txt', 'jpg')
            labels[:, 0] = 0  # reset class to 0
            with open('new/images.txt', 'a') as f:  # add image to dataset list
                f.write(img_file + '\n')
            with open('new/labels/' + Path(file).name, 'a') as f:  # write label
                for l in labels[i]:
                    f.write('%g %.6f %.6f %.6f %.6f\n' % tuple(l))
            shutil.copyfile(src=img_file, dst='new/images/' + Path(file).name.replace('txt', 'jpg'))  # copy images


def kmean_anchors(path='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            path: path to dataset *.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.general import *; _ = kmean_anchors()
    """
    thr = 1. / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1. / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x

    def fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        print('thr=%.2f: %.4f best possible recall, %.2f anchors past thr' % (thr, bpr, aat))
        print('n=%g, img_size=%s, metric_all=%.3f/%.3f-mean/best, past_thr=%.3f-mean: ' %
              (n, img_size, x.mean(), best.mean(), x[x > thr].mean()), end='')
        for i, x in enumerate(k):
            print('%i,%i' % (round(x[0]), round(x[1])), end=',  ' if i < len(k) - 1 else '\n')  # use in *.cfg
        return k

    if isinstance(path, str):  # *.yaml file
        with open(path) as f:
            data_dict = yaml.load(f, Loader=yaml.FullLoader)  # model dict
        from utils.datasets import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)
    else:
        dataset = path  # dataset

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print('WARNING: Extremely small objects found. '
              '%g of %g labels are < 3 pixels in width or height.' % (i, len(wh0)))
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # Kmeans calculation
    print('Running kmeans for %g anchors on %g points...' % (n, len(wh)))
    s = wh.std(0)  # sigmas for whitening
    k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    k *= s
    wh = torch.tensor(wh, dtype=torch.float32)  # filtered
    wh0 = torch.tensor(wh0, dtype=torch.float32)  # unflitered
    k = print_results(k)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.tight_layout()
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    npr = np.random
    f, sh, mp, s = fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm')  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = 'Evolving anchors with Genetic Algorithm: fitness = %.4f' % f
            if verbose:
                print_results(k)

    return print_results(k)


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=''):
    '''
    Print mutation results to evolve.txt (for use with train.py --evolve)
    '''
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if bucket:
        url = 'gs://%s/evolve.txt' % bucket
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system('gsutil cp %s .' % url)  # download evolve.txt if larger than local

    with open('evolve.txt', 'a') as f:  # append result
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)  # load unique rows
    x = x[np.argsort(-fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :7])
        c = '%10.4g' * len(results) % results  # results (P, R, mAP@0.5, mAP@0.5:0.95, val_losses x 3)
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system('gsutil cp evolve.txt %s gs://%s' % (yaml_file, bucket))  # upload


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):  # per image
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4])  # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = b[:, 2:] * 1.3 + 30  # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):  # per item
                cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
                im = cv2.resize(cutout, (224, 224))  # BGR
                # cv2.imwrite('test%i.jpg' % j, cutout)

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)  # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]  # retain matching class detections

    return x


def fitness(x):
    # Returns fitness (for use with results.txt or evolve.txt)
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)


def output_to_target(output, width, height):
    '''
    Convert model output to target format [batch_id, class_id, x, y, w, h, θ, conf]
    @param output: (batch_size, num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]  真实xywh
    @param width: width
    @param height: height
    @return:
            targets: (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, θ , conf] 归一化的xywh
    '''
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    targets = []
    for i, o in enumerate(output):
        if o is not None:  # o.size = (num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
            for pred in o:  # pred.size = [xywhθ,conf,classid]
                box = pred[:4]
                w = box[2] / width
                h = box[3] / height
                x = box[0] / width
                y = box[1] / height
                conf = pred[5]
                cls = int(pred[6])

                targets.append([i, cls, x, y, w, h, pred[4], conf])

    return np.array(targets)


def increment_dir(dir, comment=''):
    # Increments a directory runs/exp1 --> runs/exp2_comment
    n = 0  # number
    dir = str(Path(dir))  # os-agnostic
    d = sorted(glob.glob(dir + '*'))  # directories
    if len(d):
        n = max([int(x[len(dir):x.rfind('_') if '_' in Path(x).name else None]) for x in d]) + 1  # increment
    return dir + str(n) + ('_' + comment if comment else '')


# Plotting functions ---------------------------------------------------------------------------------------------------
def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    '''
    Plots one bounding box on image img
    @param x: [tensor(x1),tensor(y1),tensor(x2),tensor(y2)]
    @param img: 原始图片 shape=(size1,size2,3)
    @param color: size(3)   eg:[25, 184, 176]
    @param label: 字符串
    @param line_thickness: 框的厚度
    '''
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def longsideformat2poly(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) θ ∈ [0-179]    to  poly
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [0, 180)
    @return: poly shape(8)
    '''
    # Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, (theta_longside - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.double(cv2.boxPoints(rect))  # 返回rect对应的四个点的值 normalized
    poly.shape = 8
    return poly

def cvminAreaRect2longsideformat(x_c, y_c, width, height, theta):
    '''
    trans minAreaRect(x_c, y_c, width, height, θ) to longside format(x_c, y_c, longside, shortside, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param width: x轴逆时针旋转碰到的第一条边
    @param height: 与width不同的边
    @param theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    @return:
            x_c: center_x
            y_c: center_y
            longside: 最长边
            shortside: 最短边
            theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    '''
    '''
    意外情况:(此时要将它们恢复符合规则的opencv形式：wh交换，Θ置为-90)
    竖直box：box_width < box_height  θ=0
    水平box：box_width > box_height  θ=0
    '''
    if theta == 0:
        theta = -90
        buffer_width = width
        width = height
        height = buffer_width

    if theta > 0:
        if theta != 90:  # Θ=90说明wh中有为0的元素，即gt信息不完整，无需提示异常，直接删除
            print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if theta < -90:
        print('θ计算出现异常，当前数据为：%.16f, %.16f, %.16f, %.16f, %.1f;超出opencv表示法的范围：[-90,0)' % (x_c, y_c, width, height, theta))
        return False

    if width != max(width, height):  # 若width不是最长边
        longside = height
        shortside = width
        theta_longside = theta - 90
    else:  # 若width是最长边(包括正方形的情况)
        longside = width
        shortside = height
        theta_longside = theta

    if longside < shortside:
        print('旋转框转换表示形式后出现问题：最长边小于短边;[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False
    if (theta_longside < -180 or theta_longside >= 0):
        print('旋转框转换表示形式时出现问题:θ超出长边表示法的范围：[-180,0);[%.16f, %.16f, %.16f, %.16f, %.1f]' % (x_c, y_c, longside, shortside, theta_longside))
        return False

    return x_c, y_c, longside, shortside, theta_longside

def longsideformat2cvminAreaRect(x_c, y_c, longside, shortside, theta_longside):
    '''
    trans longside format(x_c, y_c, longside, shortside, θ) to minAreaRect(x_c, y_c, width, height, θ)
    两者区别为:
            当opencv表示法中width为最长边时（包括正方形的情况），则两种表示方法一致
            当opencv表示法中width不为最长边 ，则最长边表示法的角度要在opencv的Θ基础上-90度
    @param x_c: center_x
    @param y_c: center_y
    @param longside: 最长边
    @param shortside: 最短边
    @param theta_longside: 最长边和x轴逆时针旋转的夹角，逆时针方向角度为负 [-180, 0)
    @return: ((x_c, y_c),(width, height),Θ)
            x_c: center_x
            y_c: center_y
            width: x轴逆时针旋转碰到的第一条边最长边
            height: 与width不同的边
            theta: x轴逆时针旋转与width的夹角，由于原点位于图像的左上角，逆时针旋转角度为负 [-90, 0)
    '''
    if ((theta_longside >= -180) and (theta_longside < -90)):  # width is not the longest side
        width = shortside
        height = longside
        theta = theta_longside + 90
    else:
        width = longside
        height =shortside
        theta = theta_longside

    if (theta < -90) or (theta >= 0):
        print('当前θ=%.1f，超出opencv的θ定义范围[-90, 0)' % theta)

    return ((x_c, y_c), (width, height), theta)

def plot_one_rotated_box(rbox, img, color=None, label=None, line_thickness=None, pi_format=True):
    '''
    Plots one rotated bounding box on image img
    @param rbox:[tensor(x),tensor(y),tensor(l),tensor(s),tensor(θ)]
    @param img: 原始图片 shape=(size1,size2,3)
    @param color: size(3)   eg:[25, 184, 176]
    @param label: 字符串
    @param line_thickness: 框的厚度
    @param pi_format: θ是否为pi且 θ ∈ [-pi/2,pi/2)  False说明 θ∈[0,179]
    '''
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    #rbox = np.array(x)
    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]

    # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(rbox[0], rbox[1], rbox[2], rbox[3], (rbox[4] - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值
    poly = np.int0(poly)
    # 画出来
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=color, thickness=2*tl)
    c1 = (int(rbox[0]), int(rbox[1]))
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 4, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def plot_wh_methods():  # from utils.general import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOv5 ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOv5 ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.tight_layout()
    fig.savefig('comparison.png', dpi=200)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=4):
    """
    将batch中的图片绘制在一张图中
    @param images: torch.Size([batch_size, 3, weights, heights])
    @param targets: torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, θ , conf(maybe)]
    @param paths: List['img1_path','img2_path',......,'img-1_path']  len(paths)=batch_size
    @param fname: save_filename
    @param max_subplots: 一张图中最多绘制的图片数量（最多在一张图中绘制batch_size张图片 or max_subplots张图片）

    @return: mosaic 将该batch的图片绘制在一张图中（绘制的图片数量由max_subplots确定）
    """
    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise 将归一化的图像还原
    if np.max(images[0]) <= 1:
        images *= 255

    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images  保存的图中最多一次性绘制batch_size张图（不超过max_subplots=16）
    ns = np.ceil(bs ** 0.5)  # number of subplots (square) 比如batch_size为4则 subplots为2 （2*2=4）

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    # Empty array for output
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)

    # Fix class - colour map
    prop_cycle = plt.rcParams['axes.prop_cycle']
    # https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
    color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]

    # torch.Size([batch_size, 3, weights, heights])
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img
        # torch.Size = (该batch中的目标数量, [该image属于该batch的第几个图片, class, xywh, θ , conf(maybe)]
        if len(targets) > 0:
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            gt = image_targets.shape[1] == 7  # ground truth if no conf column
            theta = image_targets[:, 6]  # numpy.size=(num)  -> (num, 1)
            theta = theta[:, None]
            conf = None if gt else image_targets[:, 7]  # check for confidence presence (gt vs pred)

            boxes[[0, 2]] *= w
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] *= h
            boxes[[1, 3]] += block_y

            boxes = xyxy2xywh(boxes.T)  # numpy.size=(num, [xywh])
            rboxes = np.hstack((boxes, theta))  # numpy.size=(num, [xywhθ])
            for j, rbox in enumerate(rboxes):
                cls = int(classes[j])
                color = color_lut[cls % len(color_lut)]
                cls = names[cls] if names else cls
                if gt or conf[j] > 0.3:  # 0.3 conf thresh
                    label = '%s' % cls if gt else '%s %.1f' % (cls, conf[j])
                    #plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)
                    plot_one_rotated_box(rbox, mosaic, label=label, color=color, line_thickness=tl,
                                         pi_format=False)

        # Draw image filename labels
        if paths is not None:
            label = os.path.basename(paths[i])[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 4, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 4, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname is not None:
        mosaic = cv2.resize(mosaic, (int(ns * w * 0.5), int(ns * h * 0.5)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))

    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)


def plot_test_txt():  # from utils.general import *; plot_test()
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():  # from utils.general import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(f='study.txt', x=None):  # from utils.general import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in ['study/study_coco_yolov5%s.txt' % x for x in ['s', 'm', 'l', 'x']]:
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        for i in range(7):
            ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
            ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, :j], y[3, :j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=Path(f).stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid()
    ax2.set_xlim(0, 30)
    ax2.set_ylim(28, 50)
    ax2.set_yticks(np.arange(30, 55, 5))
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig('study_mAP_latency.png', dpi=300)
    plt.savefig(f.replace('.txt', '.png'), dpi=300)


def plot_labels(labels, save_dir=''):
    # plot dataset labels
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes

    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_xlabel('classes')
    ax[1].scatter(b[0], b[1], c=hist2d(b[0], b[1], 90), cmap='jet')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[2].scatter(b[2], b[3], c=hist2d(b[2], b[3], 90), cmap='jet')
    ax[2].set_xlabel('width')
    ax[2].set_ylabel('height')
    plt.savefig(Path(save_dir) / 'labels.png', dpi=200)
    plt.close()

    # seaborn correlogram
    try:
        import seaborn as sns
        import pandas as pd
        x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])
        sns.pairplot(x, corner=True, diag_kind='hist', kind='scatter', markers='o',
                     plot_kws=dict(s=3, edgecolor=None, linewidth=1, alpha=0.02),
                     diag_kws=dict(bins=50))
        plt.savefig(Path(save_dir) / 'labels_correlogram.png', dpi=200)
        plt.close()
    except Exception as e:
        pass


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.general import *; plot_evolution()
    '''
    Plot hyperparameter evolution results in evolve.txt
    '''
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def plot_results_overlay(start=0, stop=0):  # from utils.general import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket='', id=(), labels=(), save_dir=''):
    # from utils.general import *; plot_results()
    # Plot training 'results*.txt' as seen in https://github.com/ultralytics/yolov5#reproduce-our-training
    fig, ax = plt.subplots(2, 5, figsize=(12, 6))
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall',
         'val Box', 'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        # os.system('rm -rf storage.googleapis.com')
        # files = ['https://storage.googleapis.com/%s/results%g.txt' % (bucket, x) for x in id]
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp ' + '%s ' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = glob.glob(str(Path(save_dir) / 'results*.txt')) + glob.glob('../../Downloads/results*.txt')
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else Path(f).stem
                ax[i].plot(x, y, marker='.', label=label, linewidth=1, markersize=6)
                ax[i].set_title(s[i])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
