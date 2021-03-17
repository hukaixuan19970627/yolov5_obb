import argparse
import logging
import math
import sys
from copy import deepcopy
from pathlib import Path

sys.path.append('./')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

import torch
import torch.nn as nn

from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, Concat, NMS
from models.experimental import MixConv2d, CrossConv, C3
from utils.general import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)

class Detect(nn.Module):  # 定义检测网络
    '''
    input:(number_classes, anchors=(), ch=(tensor_small,tensor_medium,tensor_large))    tensor[i]:(batch_size, in_channels, size1, size2)
    size1[i] = img_size1/(8*i)  size2[i] = img_size2/(8*i)   eg:  tensor_small:(batch_size, inchannels, img_size1/8. img_size2/8)
    '''
    stride = None  # strides computed during build
    export = False  # onnx export,网络模型输出为onnx格式，可在其他深度学习框架上运行

    def __init__(self, nc=16, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.angle = 180
        self.no = nc + 5 + self.angle  # number of outputs per anchor   （xywh + score + num_classes + num_angle）
        self.nl = len(anchors)  # number of detection layers  3  三种步长的检测网络
        self.na = len(anchors[0]) // 2  # number of anchors 6//2=3  每种网络3种anchor框
        self.grid = [torch.zeros(1)] * self.nl  # init grid   [tensor([0.]), tensor([0.]), tensor([0.])] 初始化网格
        # anchor.shape= (3 , 6) -> shape= ( 3 , 3 , 2)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)  # shape(3, ?(3), 2)
        # register_buffer用法：内存中定义一个常量，同时，模型保存和加载的时候可以写入和读出
        self.register_buffer('anchors', a)  # shape(nl,na,2) = (3, 3, 2)
        # shape(3, 3, 2) -> shape(3, 1, 3, 1, 1, 2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,？(na),1,1,2) = (3, 1, 3, 1, 1, 2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        '''
        m(
            (0) :  nn.Conv2d(in_ch[0]（17）, (nc + 5 + self.angle) * na, kernel_size=1)  # 每个锚框中心点有3种尺度的anchor，每个anchor有 no 个输出
            (1) :  nn.Conv2d(in_ch[1]（20）, (nc + 5 + self.angle) * na, kernel_size=1)
            (2) :  nn.Conv2d(in_ch[2]（23）, (nc + 5 + self.angle) * na, kernel_size=1)
        )
        '''

    def forward(self, x):
        '''
        相当于最后生成的feature map分辨率为size1 × size2.即映射到原图，有size1 × size2个锚点，以锚点为中心生成锚框来获取Region proposals，每个锚点代表一个[xywh,score,num_classes]向量
        forward(in_tensor)   in_tensor:[(P3/8-small), (P4/16-medium), (P5/32-large)]   (3种size的featuremap, batch_size, no * na , size_1, size2)
        return :
             if training : x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
             else : (z,x)
                    z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), no) 真实坐标
                    x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
        '''
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):  # nl = 3    in:(batch_size, no * na, size1, size2)
            # x[i].shape(batch_size , (5+nc+180) * na, size1/8*(i+1) , size2/8*(i+1))
            x[i] = self.m[i](x[i])  # conv  yolo_out[i] 对各size的feature map分别进行head检测 small medium large
            # ny为featuremap的height， nx为featuremap的width
            bs, _, ny, nx = x[i].shape  # x[i]:(batch_size, (5+nc+180) * na, size1', size2')

            # x(batch_size,(5+nc+180) * 3,size1',size2') to x(batch_size,3种框,(5+nc+180),size1',size2')
            # x(batch_size,3种框,(5+nc+180),size1',size2') to x(batch_size, 3种框, size1', size2', (5+nc+180))
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference推理模式
                # grid[i].shape[2:4]=[size1, size2]  即[height/8*i, width/8*i] 与对应的featuremap层尺度一致
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    # grid[i]: tensor.shape(1, 1,当前featuremap的height, 当前featuremap的width, 2)
                    # 以height为y轴，width为x轴的grid坐标 坐标按顺序（0, 0） （1, 0）... (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                # y:(batch_size,3种scale框,size1,size2,[xywh,score,num_classes,num_angle])
                y = x[i].sigmoid()
                # i : 0为small_forward 1为medium_forward 2为large_forward
                # self.grid[i]: tensor.shape(1, 1,当前featuremap的height, 当前featuremap的width, 2) 以height为y轴，width为x轴的grid坐标
                # grid坐标按顺序（0, 0） （1, 0）...  (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
                # self.stride = ([ 8., 16., 32.])
                # self.anchor_grid: shape(3, 1, 3, 1, 1, 2)
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy 预测的真实坐标 y[..., 0:2] * 2. - 0.5是相对于左上角网格的偏移量； self.grid[i]是网格坐标索引
                # anchor_grid[i].shape=(1, 3, 1, 1, 2)  y[..., 2:4].shape=(bs, 3, height', width', 2)
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 预测的真实wh  self.anchor_grid[i]是原始anchors宽高  (y[..., 2:4] * 2) ** 2 是预测出的anchors的wh倍率
                z.append(y.view(bs, -1, self.no))  # z:(batch_size, 累加3*size1*size2 , (5+nc+180)) z会一直在[1]维度上增添数据

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):  # 绘制网格
        """
        绘制网格 eg：640 × 480的图像在detect层第一层中featuremap大小为 80 × 60，此时要生成 80 × 60的网格在原图上
        @param nx: 当前featuremap的width
        @param ny: 当前featuremap的height
        @return: tensor.shape(1, 1, 当前featuremap的height, 当前featuremap的width, 2) 生成以height为y轴，width为x轴的grid坐标
                 坐标按顺序（0, 0） （1, 0）...  (width-1, 0) (0, 1) (1,1) ... (width-1, 1) ... (width-1 , height-1)
        """
        # 初始化ny行 × nx列的tensor
        '''
        eg:  初始化ny=80行 × nx=64列的tensor
            yv = tensor([[ 0,  0,  0,  ...,  0,  0,  0],               xv = tensor([[ 0,  1,  2,  ..., 61, 62, 63],
                         [ 1,  1,  1,  ...,  1,  1,  1],                            [ 0,  1,  2,  ..., 61, 62, 63],
                         ...,                                                       ...,
                         [79, 79, 79,  ..., 79, 79, 79]])                           [ 0,  1,  2,  ..., 61, 62, 63]])
        '''
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        # 将两个 ny×ny 和 nx×nx的tensor在dim=2的维度上进行堆叠 shape(ny, nx, 2)
        '''
        eg: tensor([[
                     [ 0,  0],       [[ 0,  1],       [[ 0,  2],                 [[ 0, 79],
                     [ 1,  0],        [ 1,  1],        [ 1,  2],                  [ 1, 79],
                     ...,             ...,             ...,            ...,       ...,
                     [63,  0]],       [63,  1]],       [63,  2]],                 [63, 79]
                                                                                             ]])
        '''
        # tensor.shape(ny, nx, 2) -> shape(1, 1, ny, nx, 2)
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class Model(nn.Module):
    '''
    构建成员变量self.stride = ([ 8., 16., 32.])   ；
    更改Detect类的成员变量anchors; anchor.shape(3, 3, 2)  anchors: -> anchor(0,:,:)/ 8. , anchor(1,:,:)/ 16.  anchor(2,:,:)/ 32.
    Model (model, cfg_file, in_channnels, num_classes)
    model = Sequential(
                       (0): Focus(...)
                       ......
                       (24):Detect(...)
                       )
    '''
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None):  # model, input channels, number of classes
        super(Model, self).__init__()
        if isinstance(cfg, dict):  # 有预训练权重文件时cfg加载权重中保存的cfg字典内容；
            self.yaml = cfg  # model dict
        else:  # is *.yaml 没有预训练权重文件时加载用户定义的opt.cfg权重文件路径，再载入文件中的内容到字典中
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        # Define model
        if nc and nc != self.yaml['nc']:  # 字典中的nc与data.yaml中的nc不同，则以data.yaml中的nc为准
            print('Overriding model.yaml nc=%g with nc=%g' % (self.yaml['nc'], nc))
            self.yaml['nc'] = nc  # override yaml value
        # 返回（网络模型, Detect和Concat需要使用到的网络层数参数信息）
        # return： 网络模型每层的结构名序列：(nn.Sequential(*layers), [6, 4, 14, 10, 17, 20, 23])
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()  模型的最后一个函数为Detect层
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            # 此时 x.shape = (1, 3, s/8或16或32, 5+nc)  所以 x.shape[-2]=[s/8, s/16, s/32]
            # tensor: stride = ([ 8., 16., 32.])
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
            # 先将stride维度提升到(3, 1, 1) 之后进行每个维度的数据处理，使得 detect类的成员变量anchors由原图的尺度对应到最终的featuremaps尺度
            # anchor(3, 3, 2)  anchors: -> anchor(0,:,:)/ 8. , anchor(1,:,:)/ 16.  anchor(2,:,:)/ 32.
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)  # 确保anchors的元素顺序是从小物体的anchor到大物体的anchor
            # self.stride = ([ 8., 16., 32.])
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        print('')

    def forward(self, x, augment=False, profile=False):
        '''
        该函数为前向计算函数，输入向量经函数计算后，返回backbone+head+detect计算结果
        @param x: in_tensor shape(batch_size, 3, height, width)预处理后的图像
        @param augment: 默认为False
        @param profile: 是否估计Pytorch模型的FLOPs的标志位
        @return:
                if augment: (图像增强后的推理结果 , None)
                    else: (整体网络模型backbone+head+detect前向计算结果):
                            if training : x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
                            else : (z,x)
                                 tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), (5+nc+180))
                                 x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
                ->
                    if profile=True： return out_tensor
        '''
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train

    def forward_once(self, x, profile=False):
        '''
        该函数为前向计算函数，输入向量经函数计算后，返回backbone+head+detect计算结果
        @param x: 待前向传播的向量 size=(batch_size, 3, height, width)
        @param profile:  是否估计Pytorch模型的FLOPs的标志位
        @return: (整体网络模型backbone+head+detect前向计算结果):
                if training : x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
                else : (z,x)
                              z tensor: [small+medium+large_inference]  size=(batch_size, 3 * (small_size1*small_size2 + medium_size1*medium_size2 + large_size1*large_size2), no)
                              x list: [small_forward, medium_forward, large_forward]  eg:small_forward.size=( batch_size, 3种scale框, size1, size2, [xywh,score,num_classes,num_angle])
        '''
        y, dt = [], []  # outputs
        for m in self.model:
            # parser_model函数中定义的成员变量 m_.f, m_.type, m_.np = f, t, np  # 'from' index, module层名（如Detect Focus）, module对应层中的参数数量
            if m.f != -1:  # from : if not from previous layer / if current layer is concat or SPP
                # x为待concat/Detect的层网络的前向计算结果
                # 例子：m=Concat层函数 m.f = [-1, 4], x = [x,y[4]] ,即x= [上一层的前向计算结果, 第四层的前向计算结果]
                # y list：需要Concat/Detect的层数前向计算的结果 y = [None,None,None,None,第四层的前向计算结果,None,第六层的前向计算结果....]
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers


            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run ，前向计算网络每层；m不为concat/Detect时直接前向计算，否则先更改x为待计算的对应层数的前向计算结果，再进行Concat/Detect
            # m.i = 0/1/2/3/...../24; m.i表示当前第几个标准函数层
            # 把需要Concat/Detect的层数前向计算结果保存在y list中
            # 例：self.save=[6, 4, 14, 10, 17, 20, 23] ;y = [None,None,None,None,第四层的前向计算结果,None,第六层的前向计算结果......]
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x

    def _initialize_biases(self, cf=None):
        '''
        # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        '''
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             print('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):
        '''
        fuse model Conv2d() + BatchNorm2d() layers ，融合该两层模型
        在网络的推理阶段，可以将BN层的运算融合到Conv层中，减少运算量，加速推理
        '''
        print('Fusing layers... ')
        '''
        type(m) = 
        <class 'torch.nn.modules.container.Sequential'>
        <class 'models.common.Focus'>
        <class 'models.common.Conv'>
        <class 'torch.nn.modules.conv.Conv2d'>
        <class 'torch.nn.modules.activation.Hardswish'>
        ...
        '''
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):  # 如果函数层名为Conv标准卷积层，且同时 层中包含‘bn’属性名
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm 将'bn'属性删除
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def add_nms(self):  # fuse model Conv2d() + BatchNorm2d() layers
        if type(self.model[-1]) is not NMS:  # if missing NMS
            print('Adding NMS module... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
        return self

    def info(self, verbose=False):  # print model information
        model_info(self, verbose)


def parse_model(d, ch):  # model_dict, input_channels(3)
    '''
    @param d:  cfg_file/model_dict;
    @param ch: 3
    @return: (nn.Sequential(*layers), [6, 4, 14, 10, 17, 20, 23])  (网络， Concat和Detect需要使用到的网络层索引信息)
    '''

    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))  # 打印相关参数的类名
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors  6//2=3
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5) = 3*85 =255

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out  []  []  3
    '''
    从yaml文件中读取模型网络结构参数
    from : -1 代表是从上一层获得的输入;    -2表示从上两层获得的输入（head同理）
    number : module重复的次数
    module : 功能模块 common.py中定义的函数
    args : 功能函数的输入参数定义
    '''
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # 若module参数为字符串，则直接执行表达式
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                # 若arg参数为字符串，则直接执行表达式(如Flase None等)，否则直接等于数字本身（如64，128等）
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        # 模块重复次数为1时 ：n为1， 否则 ： n= （n * gd）向上取整
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain,BottleneckCSP层中Bottleneck层的个数

        # 排除concat，Unsample，Detect的情况
        if m in [Conv, Bottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP, C3]:
            # ch每次循环都会扩增[3]-> [3,80] -> [3,80,160] -> [3,80,160,160] -> '''
            c1, c2 = ch[f], args[0]  # c1 = 3， c2 = 每次module函数中的out_channels参数

            # Normal
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1.75  # exponential (default 2.0)
            #     e = math.log(c2 / ch[1]) / math.log(2)
            #     c2 = int(ch[1] * ex ** e)
            # if m != Focus:

            '''
            若c2不等于85（num_classes + 5）则 ：  c2=make_divisible(c2 * gw, 8)确保能把8整除 ；  否则：c2=c2
            '''
            c2 = make_divisible(c2 * gw, 8) if c2 != no else c2

            # Experimental
            # if i > 0 and args[0] != no:  # channel expansion factor
            #     ex = 1 + gw  # exponential (default 2.0)
            #     ch1 = 32  # ch[1]
            #     e = math.log(c2 / ch1) / math.log(2)  # level 1-n
            #     c2 = int(ch1 * ex ** e)
            # if m != Focus:
            #     c2 = make_divisible(c2, 8) if c2 != no else c2
            args = [c1, c2, *args[1:]]  # [ch[-1], out_channels, kernel_size, strides(可能)] — 除了BottleneckCSP与C3层
            if m in [BottleneckCSP, C3]:
                args.insert(2, n)       # [ch[-1], out_channnels, Bottleneck_num] — BottleneckCSP与C3层
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            # 以第一个concat为例 ： ch[-1] + ch[x+1] = ch[-1]+ch[7] = 640 + 640 = 1280
            c2 = sum([ch[-1 if x == -1 else x + 1] for x in f])
        elif m is Detect:
            args.append([ch[x + 1] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        # 构建n次的module处理模块，如构建 4次 BottleneckCSP层的模块，输入参数由args导入
        '''以第一层focus为例
        args： [ch[-1], out_channels, kernel_size, strides(可能)] = [3, 80, 3]
        m: class 'models.common.Focus'
        m_: Focus(  # focus函数会在一开始将3通道的图像再次分为12通道
                 (conv): Conv(
                              (conv): Conv2d(12, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                              (bn): BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                              (act): Hardswish()
                              )
                  )
        '''
        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        # 将'__main__.Detect'变为Detect，其余模块名不变，相当于所有函数名全都放在了t中
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        # 返回当前module结构中参数的总数目
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        # 对应相关参数的类名，打印对应参数
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        # 把Concat，Detect需要使用到的参数层的层数信息储存进save
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        # 将每层结构的函数名拓展进layers list
        layers.append(m_)
        # 将每层结构的out_channels拓展进ch，以便下一层结构调用上一层的输出通道数 yolov5.yaml中的第0层的输出对应ch[1] ;i - ch[i+1]
        ch.append(c2)
    '''
    layers=[
            Focus(...)
            Conv(...)
            ...
            Detect(...)
            ]
            
    '''
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5x.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)
    # y = model(img, profile=True)

    # ONNX export
    # model.model[-1].export = True
    # torch.onnx.export(model, img, opt.cfg.replace('.yaml', '.onnx'), verbose=True, opset_version=11)

    # Tensorboard
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter()
    # print("Run 'tensorboard --logdir=models/runs' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(model.model, img)  # add model to tensorboard
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
