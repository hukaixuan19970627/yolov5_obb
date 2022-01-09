"""
Oriented Bounding Boxes utils
"""
import numpy as np
pi = 3.141592
import cv2
import torch

def gaussian_label_cpu(label, num_class, u=0, sig=4.0):
    """
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    Args:
        label (float32):[1], theta class
        num_theta_class (int): [1], theta class num
        u (float32):[1], μ in gaussian function
        sig (float32):[1], σ in gaussian function, which is window radius for Circular Smooth Label
    Returns:
        csl_label (array): [num_theta_class], gaussian function smooth label
    """
    x = np.arange(-num_class/2, num_class/2)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))
    index = int(num_class/2 - label)
    return np.concatenate([y_sig[index:], 
                           y_sig[:index]], axis=0)

def regular_theta(theta, mode='180', start=-pi/2):
    """
    limit theta ∈ [-pi/2, pi/2)
    """
    assert mode in ['360', '180']
    cycle = 2 * pi if mode == '360' else pi

    theta = theta - start
    theta = theta % cycle
    return theta + start

def poly2rbox(polys, num_cls_thata=180, radius=6.0, use_pi=False, use_gaussian=False):
    """
    Trans poly format to rbox format.
    Args:
        polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
        num_cls_thata (int): [1], theta class num
        radius (float32): [1], window radius for Circular Smooth Label
        use_pi (bool): True θ∈[-pi/2, pi/2) ， False θ∈[0, 180)

    Returns:
        use_gaussian True:
            rboxes (array): 
            csl_labels (array): (num_gts, num_cls_thata)
        elif 
            rboxes (array): (num_gts, [cx cy l s θ]) 
    """
    assert polys.shape[-1] == 8
    if use_gaussian:
        csl_labels = []
    rboxes = []
    for poly in polys:
        poly = np.float32(poly.reshape(4, 2))
        (x, y), (w, h), angle = cv2.minAreaRect(poly) # θ ∈ [0， 90]
        angle = -angle # θ ∈ [-90， 0]
        theta = angle / 180 * pi # 转为pi制

        # trans opencv format to longedge format θ ∈ [-pi/2， pi/2]
        if w != max(w, h): 
            w, h = h, w
            theta += pi/2
        theta = regular_theta(theta) # limit theta ∈ [-pi/2, pi/2)
        angle = (theta * 180 / pi) + 90 # θ ∈ [0， 180)

        if not use_pi: # 采用angle弧度制 θ ∈ [0， 180)
            rboxes.append([x, y, w, h, angle])
        else: # 采用pi制
            rboxes.append([x, y, w, h, theta])
        if use_gaussian:
            csl_label = gaussian_label_cpu(label=angle, num_class=num_cls_thata, u=0, sig=radius)
            csl_labels.append(csl_label)
    if use_gaussian:
        return np.array(rboxes), np.array(csl_labels)
    return np.array(rboxes)

# def rbox2poly(rboxes):
#     """
#     Trans rbox format to poly format.
#     Args:
#         rboxes (array): (num_gts, [cx cy l s θ]) θ∈(0, 180]

#     Returns:
#         polys (array): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
#     """
#     assert rboxes.shape[-1] == 5
#     polys = []
#     for rbox in rboxes:
#         x, y, w, h, theta = rbox
#         if theta > 90 and theta <= 180: # longedge format -> opencv format
#             w, h = h, w
#             theta -= 90
#         if theta <= 0 or theta > 90:
#             print("cv2.minAreaRect occurs some error. θ isn't in range(0, 90]. The longedge format is: ", rbox)

#         poly = cv2.boxPoints(((x, y), (w, h), theta)).reshape(-1)  
#         polys.append(poly)
#     return np.array(polys)

def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4]) 
    """
    if isinstance(obboxes, torch.Tensor):
        center, w, h, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
        Cos, Sin = torch.cos(theta), torch.sin(theta)

        vector1 = torch.cat(
            (w/2 * Cos, -w/2 * Sin), dim=-1)
        vector2 = torch.cat(
            (-h/2 * Sin, -h/2 * Cos), dim=-1)
        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return torch.cat(
            (point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate(
            [w/2 * Cos, -w/2 * Sin], axis=-1)
        vector2 = np.concatenate(
            [-h/2 * Sin, -h/2 * Cos], axis=-1)

        point1 = center + vector1 + vector2
        point2 = center + vector1 - vector2
        point3 = center - vector1 - vector2
        point4 = center - vector1 + vector2
        order = obboxes.shape[:-1]
        return np.concatenate(
            [point1, point2, point3, point4], axis=-1).reshape(*order, 8)

def poly2hbb(polys):
    """
    Trans poly format to hbb format
    Args:
        rboxes (array/tensor): (num_gts, poly) 

    Returns:
        hbboxes (array/tensor): (num_gts, [xc yc w h]) 
    """
    assert polys.shape[-1] == 8
    if isinstance(polys, torch.Tensor):
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = torch.amax(x, dim=1) # (num)
        x_min = torch.amin(x, dim=1)
        y_max = torch.amax(y, dim=1)
        y_min = torch.amin(y, dim=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = torch.cat((x_ctr, y_ctr, w, h), dim=1)
    else:
        x = polys[:, 0::2] # (num, 4) 
        y = polys[:, 1::2]
        x_max = np.amax(x, axis=1) # (num)
        x_min = np.amin(x, axis=1) 
        y_max = np.amax(y, axis=1)
        y_min = np.amin(y, axis=1)
        x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
        h = y_max - y_min # (num)
        w = x_max - x_min
        x_ctr, y_ctr, w, h = x_ctr.reshape(-1, 1), y_ctr.reshape(-1, 1), w.reshape(-1, 1), h.reshape(-1, 1) # (num, 1)
        hbboxes = np.concatenate((x_ctr, y_ctr, w, h), axis=1)
    return hbboxes

def poly_filter(polys, h, w): 
    """
    Filter the poly labels which is out of the image.
    Args:
        polys (array): (num, 8)

    Return：
        keep_masks (array): (num)
    """
    x = polys[:, 0::2] # (num, 4) 
    y = polys[:, 1::2]
    x_max = np.amax(x, axis=1) # (num)
    x_min = np.amin(x, axis=1) 
    y_max = np.amax(y, axis=1)
    y_min = np.amin(y, axis=1)
    x_ctr, y_ctr = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0 # (num)
    keep_masks = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) 
    return keep_masks