import torch
from utils.general import longsideformat2cvminAreaRect
import cv2
import os
# -*- coding: utf-8 -*-
"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
from utils import polyiou
import copy
import cv2
import random
from PIL import Image

## the IoU thresh for nms when merge image
nms_thresh = 0.3

def py_cpu_nms_poly(dets, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly, confidence1]) 原始图像中的检测出的目标数量
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """
    scores = dets[:, 8]
    polys = []
    areas = []
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

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nameboxdict_classname, nms, thresh):
    """
    对namedict中的目标信息进行nms.不改变输入的数据形式
    @param nameboxdict: eg:{
                           'P706':[[poly1, confidence1], ..., [poly9, confidence9]],
                           ...
                           'P700':[[poly1, confidence1], ..., [poly9, confidence9]]
                            }
    @param nameboxdict_classname: eg:{
                           'P706':[[poly1, confidence1,'classname'], ..., [poly9, confidence9, 'classname']],
                           ...
                           'P700':[[poly1, confidence1, 'classname'], ..., [poly9, confidence9, 'classname']]
                            }
    @param nms:
    @param thresh: nms阈值, IoU阈值
    @return:
            nameboxnmsdict: eg:{
                                'P706':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']],
                                 ...
                                'P700':[[poly1, confidence1, 'classname'], ..., [poly_nms, confidence9, 'classname']]
                               }
    """
    # 初始化字典
    nameboxnmsdict = {x: [] for x in nameboxdict}  # eg: nameboxnmsdict={'P0770': [], 'P1888': []}
    for imgname in nameboxdict:  # 提取nameboxdict中的key eg:P0770   P1888
        keep = nms(np.array(nameboxdict[imgname]), thresh)  # rotated_nms索引值列表
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict_classname[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict

def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def custombasename(fullname):
    return os.path.basename(os.path.splitext(fullname)[0])

def GetFileFromThisRootDir(dir,ext = None):
  allfiles = []
  needExtFilter = (ext != None)
  for root,dirs,files in os.walk(dir):
    for filespath in files:
      filepath = os.path.join(root, filespath)
      extension = os.path.splitext(filepath)[1][1:]
      if needExtFilter and extension in ext:
        allfiles.append(filepath)
      elif not needExtFilter:
        allfiles.append(filepath)
  return allfiles

def mergebase(srcpath, dstpath, nms):
    """
    将源路径中所有的txt目标信息,经nms后存入目标路径中的同名txt
    @param srcpath: 合并前信息保存的txt源路径
    @param dstpath: 合并后信息保存的txt目标路径
    @param nms: NMS函数
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, name + '.txt')  # eg: example_merge/..P0001.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            nameboxdict = {}
            nameboxdict_classname = {}
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [待merge图片名(该目标所处图片名称), confidence, x1, y1, x2, y2, x3, y3, x4, y4, classname]
                subname = splitline[0]  # 每行的第一个元素 是被分割的图片的图片名 eg:P0706__1__0___0
                splitname = subname.split('__')  # 分割待merge的图像的名称 eg:['P0706','1','0','_0']
                oriname = splitname[0]  # 获得待merge图像的原图像名称 eg:P706
                pattern1 = re.compile(r'__\d+___\d+')  # 预先编译好r'__\d+___\d+' 提高重复使用效率 \d表示数字

                x_y = re.findall(pattern1, subname)  # 匹配subname中的字符串 eg: x_y=['__0___0']
                x_y_2 = re.findall(r'\d+', x_y[0])  # 匹配subname中的字符串 eg: x_y_2= ['0','0']
                x, y = int(x_y_2[0]), int(x_y_2[1])  # 找到当前subname图片在原图中的分割位置

                pattern2 = re.compile(r'__([\d+\.]+)__\d+___')  # \.表示一切字符

                rate = re.findall(pattern2, subname)[0]  # 找到该subname分割图片时的分割rate (resize rate before cut)

                confidence = splitline[1]
                classname = splitline[-1]
                poly = list(map(float, splitline[2:10]))  # 每个元素映射为浮点数 再放入列表中
                origpoly = poly2origpoly(poly, x, y, rate)  # 将目标位置信息resize 恢复成原图的poly坐标
                det = origpoly  # shape(8)
                det.append(confidence)  # [poly, 'confidence']
                det = list(map(float, det))  # [poly, confidence]

                det_classname = copy.deepcopy(det)
                det_classname.append(classname)  # [poly, 'confidence','classname']
                if (oriname not in nameboxdict):
                    nameboxdict[oriname] = []   # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1], ..., ]
                    nameboxdict_classname[oriname] = []   # 弄个元组,汇集原图目标信息 eg: 'P706':[[poly1, confidence1,'classname'], ..., ]
                nameboxdict[oriname].append(det)
                nameboxdict_classname[oriname].append(det_classname)

            nameboxnmsdict = nmsbynamedict(nameboxdict, nameboxdict_classname, nms, nms_thresh)  # 对nameboxdict元组进行nms
            with open(dstname, 'w') as f_out:
                for imgname in nameboxnmsdict:  # 'P706'
                    for det in nameboxnmsdict[imgname]:  # 取出对应图片的nms后的目标信息
                        # det:[poly1, confidence1, 'classname']
                        #print('det:', det)
                        confidence = det[-2]
                        bbox = det[0:-2]
                        outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox)) + ' ' + det[-1]
                        #print('outline:', outline)
                        f_out.write(outline + '\n')

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    @param srcpath: result files before merge and nms.txt的信息格式为:[P0770__1__0___0 confidence poly 'classname']
    @param dstpath: result files after merge and nms.保存的txt信息格式为:[P0770 confidence poly 'classname']
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'

    mergebase(srcpath,
              dstpath,
              py_cpu_nms_poly)

def rbox2txt(rbox, classname, conf, img_name, out_path, pi_format=False):
    """
    将分割图片的目标信息填入原始图片.txt中
    @param robx: rbox:[tensor(x),tensor(y),tensor(l),tensor(s),tensor(θ)]
    @param classname: string
    @param conf: string
    @param img_name: string
    @param path: 文件夹路径 str
    @param pi_format: θ是否为pi且 θ ∈ [-pi/2,pi/2)  False说明 θ∈[0,179]
    """
    if isinstance(rbox, torch.Tensor):
        rbox = rbox.cpu().float().numpy()

    #rbox = np.array(x)
    if pi_format:  # θ∈[-pi/2,pi/2)
        rbox[-1] = (rbox[-1] * 180 / np.pi) + 90  # θ∈[0,179]

    # rect=[(x_c,y_c),(w,h),Θ] Θ:flaot[0-179]  -> (-180,0)
    rect = longsideformat2cvminAreaRect(rbox[0], rbox[1], rbox[2], rbox[3], (rbox[4] - 179.9))
    # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    poly = np.float32(cv2.boxPoints(rect))  # 返回rect对应的四个点的值
    poly = np.int0(poly).reshape(8)

    splitname = img_name.split('__')  # 分割待merge的图像的名称 eg:['P0706','1','0','_0']
    oriname = splitname[0]  # 获得待merge图像的原图像名称 eg:P706

    # 目标所属图片名称_分割id 置信度 poly classname
    lines = img_name + ' ' + conf + ' ' + ' '.join(list(map(str, poly))) + ' ' + classname
    # 移除之前的输出文件夹,并新建输出文件夹
    if not os.path.exists(out_path):
        os.makedirs(out_path)  # make new output folder

    with open(str(out_path + '/' + oriname) + '.txt', 'a') as f:
        f.writelines(lines + '\n')

def evaluation_trans(srcpath, dstpath):
    """
    将srcpath文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中:
            txt中的内容格式:  目标所属原始图片名称 置信度 poly
    @param srcpath: 存放图片的目标检测结果(文件夹,内含多个txt)
                    txt中的内容格式: 目标所属图片名称 置信度 poly 'classname'
    @param dstpath: 存放图片的目标检测结果(文件夹, 内含多个Task1_类别名.txt )
                    txt中的内容格式:  目标所属原始图片名称 置信度 poly
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['result_merged/P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'result_merged/P0001.txt'
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(fullname, 'r') as f_in:
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            for splitline in splitlines:  # splitline:每行中的m个元素
                # splitline = [目标所属图片名称, confidence, x1, y1, x2, y2, x3, y3, x4, y4, 'classname']
                classname = splitline[-1]  # 每行的最后一个元素 是被分割的图片的种类名
                dstname = os.path.join(dstpath, 'Task1_' + classname + '.txt')  # eg: result/Task1_plane.txt
                lines_ = ' '.join(list(splitline[:-1]))
                with open(dstname, 'a') as f:
                    f.writelines(lines_ + '\n')

def image2txt(srcpath, dstpath):
    """
    将srcpath文件夹下的所有子文件名称打印到namefile.txt中
    @param srcpath: imageset
    @param dstpath: imgnamefile.txt的存放路径
    """
    filelist = GetFileFromThisRootDir(srcpath)  # srcpath文件夹下的所有文件相对路径 eg:['example_split/../P0001.txt', ..., '?.txt']
    for fullname in filelist:  # 'example_split/../P0001.txt'
        name = custombasename(fullname)  # 只留下文件名 eg:P0001
        dstname = os.path.join(dstpath, 'imgnamefile.txt')  # eg: result/imgnamefile.txt
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        with open(dstname, 'a') as f:
            f.writelines(name + '\n')

def draw_DOTA_image(imgsrcpath, imglabelspath, dstpath, extractclassname, thickness):
    """
    绘制工具merge后的目标/DOTA GT图像
        @param imgsrcpath: merged后的图像路径(原始图像路径)
        @param imglabelspath: merged后的labels路径
        @param dstpath: 目标绘制之后的保存路径
        @param extractclassname: the category you selected
    """
    if not os.path.exists(dstpath):
        os.makedirs(dstpath)
    # 设置画框的颜色    colors = [[178, 63, 143], [25, 184, 176], [238, 152, 129],....,[235, 137, 120]]随机设置RGB颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(extractclassname))]
    filelist = GetFileFromThisRootDir(imglabelspath)  # fileist=['/.../P0005.txt', ..., /.../P000?.txt]
    for fullname in filelist:  # fullname='/.../P000?.txt'
        objects = []
        with open(fullname, 'r') as f_in:  # 打开merge后/原始的DOTA图像的gt.txt
            lines = f_in.readlines()  # 读取txt中所有行,每行作为一个元素存于list中
            splitlines = [x.strip().split(' ') for x in lines]  # 再次分割list中的每行元素 shape:n行 * m个元素
            if len(splitlines[0]) == 1:  # 首行为"imagesource:GoogleEarth",说明为DOTA原始labels
                # DOTA labels:[polys classname 1/0]
                del splitlines[0]
                del splitlines[0]   # 删除前两个无用信息
                objects = [x[0:-2] for x in splitlines]
                classnames = [x[-2] for x in splitlines]
            else:
                # P0003 0.911 660.0 309.0 639.0 209.0 661.0 204.0 682.0 304.0 large-vehicle
                objects = [x[2:-1] for x in splitlines]
                classnames = [x[-1] for x in splitlines]

        '''
        objects[i] = str[poly, classname]
        '''
        name = os.path.splitext(os.path.basename(fullname))[0]  # name='P000?'
        img_fullname = os.path.join(imgsrcpath, name + '.png')
        img_savename = os.path.join(dstpath, name + '_.png')
        img = cv2.imread(img_fullname)  # 读取图像像素

        for i, obj in enumerate(objects):
            # obj = [poly ,'classname']
            classname = classnames[i]
            # poly = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
            poly = np.array(list(map(float, obj)))
            poly = poly.reshape(4, 2)  # 返回rect对应的四个点的值 normalized
            poly = np.int0(poly)

            # 画出来
            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(extractclassname.index(classname))],
                             thickness=thickness)
        cv2.imwrite(img_savename, img)







if __name__ == '__main__':
    '''
        计算AP的流程:
        1.detect.py检测一个文件夹的所有图片并把检测结果按照图片原始来源存入 原始图片名称.txt中:   (rbox2txt函数)
            txt中的内容格式:  目标所属图片名称_分割id 置信度 poly classname
        2.ResultMerge.py将所有 原始图片名称.txt 进行merge和nms,并将结果存入到另一个文件夹的 原始图片名称.txt中: (mergebypoly函数)
            txt中的内容格式:  目标所属图片名称 置信度 poly classname
        3.写一个evaluation_trans函数将上个文件夹中的所有txt中的目标提取出来,按照目标类别分别存入 Task1_类别名.txt中:
            txt中的内容格式:  目标所属原始图片名称 置信度 poly
        4.写一个imgname2txt.py 将测试集的所有图片名称打印到namefile.txt中
    '''
    # see demo for example
    mergebypoly(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_before_merge',
                r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged')
    print('检测结果已merge')
    evaluation_trans(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged',
                     r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_classname')
    print('检测结果已按照类别分类')
    image2txt(r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',  # val原图数据集路径
              r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt')
    print('校验数据集名称文件已生成')

    # classnames_v1_5 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle',
    #                    'ship', 'tennis-court',
    #                    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool',
    #                    'helicopter', 'container-crane']
    #
    # draw_DOTA_image(imgsrcpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/row_images',
    #                 imglabelspath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/result_txt/result_merged',
    #                 dstpath=r'/home/test/Persons/hukaixuan/yolov5_DOTA_OBB/DOTA_demo_view/detection/merged_drawed',
    #                 extractclassname=classnames_v1_5,
    #                 thickness=2
    #                 )