#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
from torch import nn
import sys
sys.path.append(r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3')
from models.Yolofastestrevise import *
import torchvision

def intialization(model):
    '''https://github.com/Peterisfar/YOLOV3/blob/master/model/yolov3.py#L68'''
    for name, m in model.named_children(): #1 
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
            print("initialized: {}".format(m))
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)
            print("initialized: {}".format(m))
        else:
            intialization(m)

def loadweight(weights, count, m, ptr):
    # if count == cutoff:
    #     return None
    # count += 1
    layers = []
    try:
        norm1 = m.norm1
        layers.append(norm1)
    except:
        pass
    try:
        conv1 = m.conv1
        layers.append(conv1)
    except:
        pass
    try:
        norm2 = m.norm2
        layers.append(norm2)
    except:
        pass
    try:
        conv2 = m.conv2
        layers.append(conv2)
    except:
        pass
    try:
        norm_extra = m.norm_extra
        layers.append(norm_extra)
    except:
        pass
    try:
        conv_extra = m.conv_extra
        layers.append(conv_extra)
    except:
        pass
    for mn in layers:
        conv_layer = mn
        if isinstance(mn, nn.BatchNorm2d):
            # Load BN bias, weights, running mean and running variance
            bn_layer = mn
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b

            print("loaded {}".format(bn_layer))
        else:
            # Load conv. bias
            if conv_layer.bias!=None:
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
            print("loaded {}".format(conv_layer))

    return count, ptr

def load_darknet_weights(model: torch.nn.Module, weight_file, cutoff=52):
    "https://github.com/ultralytics/yolov3/blob/master/models.py"

    print("load darknet weights : ", weight_file)

    with open(weight_file, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    count = 0
    ptr = 0

    for name, m in model.named_children(): #1 
        if isinstance(m, inputnet):
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock1):
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock2):
            for i in range(len(m.resblock2_child)):
                count, ptr = loadweight(weights, count, m.resblock2_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock3):
            for i in range(len(m.resblock3_child)):
                count, ptr = loadweight(weights, count, m.resblock3_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock4):
            for i in range(len(m.resblock4_child)):
                count, ptr = loadweight(weights, count, m.resblock4_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock5):
            for i in range(len(m.resblock5_child)):
                count, ptr = loadweight(weights, count, m.resblock5_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)

def freeze_darknet_backbone(model):
    '''https://github.com/qqwweee/keras-yolo3/blob/master/train.py#L50'''
    freeze_netlist = ["net", "ResBlock1", "ResBlock2", "ResBlock3", "ResBlock4", "ResBlock5"]
    for name, m in model.named_parameters(): #1 
        mar = 999
        for i in freeze_netlist:
            if i in name:
                mar = -999
                break
        if mar < 0:
            print("freeze layer params: ", name, list(m.size()))
            m.requires_grad = False
        else:
            print("train layer params: ", name, list(m.size()))
            continue

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    conf_thres = 0.1
    xc = prediction[..., 4] > conf_thres  # candidates
    # print(prediction[prediction[..., 4] > conf_thres], conf_thres, prediction[..., 4])
    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords
