#Author：ZouJiu
#Time: 2021-8-13

from cProfile import label
import os
import cv2
import time
import torch
from copy import deepcopy 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class inputnet(nn.Module):
    def __init__(self):
        super(inputnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
       x = self.leaky_relu(self.norm1(self.conv1(x)))
       x = self.leaky_relu(self.norm2(self.conv2(x)))
       return x

class resblock1(nn.Module):
    def __init__(self):
        super(resblock1, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.relu = nn.ReLU()
        
        self.conv_extra = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.norm_extra = nn.BatchNorm2d(128)


    def forward(self, x):
        y = self.leaky_relu(self.norm1(self.conv1(x)))
        y = self.leaky_relu(self.norm2(self.conv2(y)))
        y = x + y
        y = self.leaky_relu(self.norm_extra(self.conv_extra(y)))

        return y

class resblock2_child(nn.Module):
    def __init__(self):
        super(resblock2_child, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(128)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu(self.norm1(self.conv1(x)))
        y = self.leaky_relu(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock2(nn.Module):
    def __init__(self):
        super(resblock2, self).__init__()
        self.resblock2_child = nn.ModuleList([resblock2_child() for i in range(2)])
        self.conv_extra = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.norm_extra = nn.BatchNorm2d(256)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        for i, rb2 in enumerate(self.resblock2_child):
            x = rb2(x)
        x = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
        return x

class resblock3_child(nn.Module):
    def __init__(self):
        super(resblock3_child, self).__init__()
        self.conv1 = nn.Conv2d(256, 128, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(256)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu(self.norm1(self.conv1(x)))
        y = self.leaky_relu(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock3(nn.Module):
    def __init__(self):
        super(resblock3, self).__init__()
        self.resblock3_child = nn.ModuleList([resblock3_child() for i in range(8)])
        self.conv_extra = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.norm_extra = nn.BatchNorm2d(512)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        for i, rb2 in enumerate(self.resblock3_child):
            x = rb2(x)
        y = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
        return x,y

class resblock4_child(nn.Module):
    def __init__(self):
        super(resblock4_child, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(512)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu(self.norm1(self.conv1(x)))
        y = self.leaky_relu(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock4(nn.Module):
    def __init__(self):
        super(resblock4, self).__init__()
        self.resblock4_child = nn.ModuleList([resblock4_child() for i in range(8)])
        self.conv_extra = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)
        self.norm_extra = nn.BatchNorm2d(1024)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x):
        for i, rb2 in enumerate(self.resblock4_child):
            x = rb2(x)
        y = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
        return x, y

class resblock5_child(nn.Module):
    def __init__(self):
        super(resblock5_child, self).__init__()
        self.conv1 = nn.Conv2d(1024, 512, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(1024)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu(self.norm1(self.conv1(x)))
        y = self.leaky_relu(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock5(nn.Module):
    def __init__(self):
        super(resblock5, self).__init__()
        self.resblock5_child = nn.ModuleList([resblock5_child() for i in range(4)])

    def forward(self, x):
        for i, rb2 in enumerate(self.resblock5_child):
            x = rb2(x)
        return x

class header_block(nn.Module):
    def __init__(self, num_classes, beginchannel, channel, upornot=True):
        super(header_block, self).__init__()
        self.upornot = upornot
        self.conv1 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(beginchannel)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv3 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(beginchannel)
        self.conv5 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
        self.norm5 = nn.BatchNorm2d(channel)

        #yolo1,接conv5
        self.conv7 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
        self.norm7 = nn.BatchNorm2d(beginchannel)
        self.conv8 = nn.Conv2d(beginchannel, (5+num_classes)*3, 1, stride=1, bias=False)
        self.relu = nn.ReLU()

        #upsample,接conv5
        self.conv9 = nn.Conv2d(channel, channel//2, 1, stride=1, bias=False)
        self.norm9 = nn.BatchNorm2d(channel//2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.leaky_relu(self.norm1(self.conv1(x)))
        x = self.leaky_relu(self.norm2(self.conv2(x)))
        x = self.leaky_relu(self.norm3(self.conv3(x)))
        x = self.leaky_relu(self.norm4(self.conv4(x)))
        x = self.leaky_relu(self.norm5(self.conv5(x)))

        y3 = self.leaky_relu(self.norm7(self.conv7(x)))
        y3 = self.conv8(y3)
        if self.upornot:
            up3 = self.leaky_relu(self.norm9(self.conv9(x)))
            up3 = self.upsample(up3)
            return up3, y3
        else:
            return y3

def iou_p_g(groundtruth, predict):
    cx, cy, w, h = groundtruth[:, 0], groundtruth[:, 1],groundtruth[:, 2],groundtruth[:, 3]
    cxp, cyp, wp, hp = predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3]
    xmin = cx - w/2
    ymin = cy - h/2
    xmax = cx + w/2
    ymax = cy + h/2

    xminp = cxp - wp/2
    yminp = cyp - hp/2
    xmaxp = cxp + wp/2
    ymaxp = cyp + hp/2
    alliou = torch.zeros((predict.size()[0], groundtruth.size()[0]))
    for i in range(groundtruth.size()[0]):
        singleiou = []
        join = (torch.min(xmax[i], xmaxp) - torch.max(xmin[i], xminp)).clamp(0)*\
            (torch.min(ymax[i], ymaxp) - torch.max(ymin[i], yminp)).clamp(0)
        gtarea = (xmax[i] - xmin[i])*(ymax[i] - ymin[i])
        parea = (xmaxp - xminp)*(ymaxp - yminp)
        iouresult = join/(gtarea+parea-join)
        alliou[:, i] = iouresult
    return alliou

def iou_box(groundtruth, predict):
    cx, cy, w, h = groundtruth[:, 0], groundtruth[:, 1],groundtruth[:, 2],groundtruth[:, 3]
    cxp, cyp, wp, hp = predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3]
    xmin = cx - w/2
    ymin = cy - h/2
    xmax = cx + w/2
    ymax = cy + h/2

    xminp = cxp - wp/2
    yminp = cyp - hp/2
    xmaxp = cxp + wp/2
    ymaxp = cyp + hp/2

    join = (torch.min(xmax, xmaxp) - torch.max(xmin, xminp))*\
        (torch.min(ymax, ymaxp) - torch.max(ymin, yminp))
    gtarea = (xmax - xmin)*(ymax - ymin)
    parea = (xmaxp - xminp)*(ymaxp - yminp)
    iouresult = join/(gtarea+parea-join)
    return iouresult

def iouwh(batch_gt00wh, anchor00wh): 
    #定义两者的中心点坐标重合，只要考虑长和宽，不妨定中心点(x, y) = (90, 90)，实际不会是0
    #做了inner全连接
    assume = torch.ones(batch_gt00wh.size()[0])*90
    if torch.cuda.is_available():
        assume = assume.to("cuda")
    gtw, gth = batch_gt00wh[:, 0], batch_gt00wh[:, 1]
    anchorw, anchorh = anchor00wh[:, 0], anchor00wh[:, 1]
    xmingt = assume - gtw/2
    ymingt = assume - gth/2
    xmaxgt = assume + gtw/2
    ymaxgt = assume + gth/2

    xminan = assume - anchorw/2
    yminan = assume - anchorh/2
    xmaxan = assume + anchorw/2
    ymaxan = assume + anchorh/2
    join = (torch.min(xmaxgt, xmaxan) - torch.max(xmingt, xminan)).clamp(0)*\
        (torch.min(ymaxgt, ymaxan) - torch.max(ymingt, yminan)).clamp(0)
    gtarea = (xmaxgt - xmingt)*(ymaxgt - ymingt)
    anchorarea = (xmaxan - xminan)*(ymaxan - yminan)
    iouresult = join/(gtarea+anchorarea-join)
    length = iouresult.size()[0]//9
    #对全连接进行合并
    res1 = torch.unsqueeze(iouresult[:length], 1)
    res2 = torch.unsqueeze(iouresult[length:length*2], 1)
    res3 = torch.unsqueeze(iouresult[length*2:length*3], 1)
    res4 = torch.unsqueeze(iouresult[length*3:length*4], 1)
    res5 = torch.unsqueeze(iouresult[length*4:length*5], 1)
    res6 = torch.unsqueeze(iouresult[length*5:length*6], 1)
    res7 = torch.unsqueeze(iouresult[length*6:length*7], 1)
    res8 = torch.unsqueeze(iouresult[length*7:length*8], 1)
    res9 = torch.unsqueeze(iouresult[length*8:], 1)
    result = torch.cat([res1, res2, res3,\
        res4, res5, res6,\
        res7, res8, res9], dim=1) #[gt的数量, 3]，3代表3个anchor，也就是特征图的3层
    return result

def nms(predict, nms_thresh):
    #[cx, cy, w, h, maxscore, label]
    if len(predict)==0:
        return []
    index  = np.argsort(predict[:, 4])
    index = list(index)
    index.reverse()
    predict = predict[index]
    xmin = predict[:, 0] - predict[:, 2]/2
    ymin = predict[:, 1] - predict[:, 3]/2
    xmax = predict[:, 0] + predict[:, 2]/2
    ymax = predict[:, 1] + predict[:, 3]/2
    areas = (ymax - ymin)*(xmax - xmin)
    labeles = np.unique(predict[:, 5])
    keep = []
    # print(predict, predict.shape)
    for j in range(len(labeles)):
        ind = np.where(predict[:, 5]==labeles[j])[0]
        if len(ind)==0:
            continue
        # if len(ind)!=1:
        #     print(ind)
        while len(ind)>0:
            i = ind[0]
            keep.append(i)

            x1min = np.maximum(xmin[i], xmin[ind[1:]])
            y1min = np.maximum(ymin[i], ymin[ind[1:]])
            x1max = np.minimum(xmax[i], xmax[ind[1:]])
            y1max = np.minimum(ymax[i], ymax[ind[1:]])
            overlap = np.maximum(0, (y1max-y1min))*np.maximum(0, (x1max-x1min))

            ioures = overlap/(areas[i] + areas[ind[1:]] - overlap)
            # t = np.where(ioures <= nms_thresh)[0]
            maskiou = ioures<= nms_thresh
            if len(maskiou)==0:
                break
            # print(1111111, ind)
            ind = ind[1:][ioures <= nms_thresh]
            # print(ioures <= nms_thresh, ind)
    # print(3333333, keep)
    return predict[keep]

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

def GIOU_xywh_torch(boxes1, boxes2):
    """
     https://arxiv.org/abs/1902.09630
     https://github.com/Peterisfar/YOLOV3/blob/master/utils/tools.py#L199
    boxes1(boxes2)' shape is [..., (x,y,w,h)].The size is for original image.
    """
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area =  inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10
    IOU = 1.0 * inter_area / union_area
    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
    # print(9999999999, IOU, enclose_area, union_area, enclose_area)
    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    return IOU


def iou_xywh_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area + 1e-10
    IOU = 1.0 * inter_area / union_area
    return IOU

class yolo(nn.Module):
    '''
    对输出的特征图进行解码，得到坐标、置信度、分类的概率
    并计算相应的objectness loss、classify loss、coordinates loss
    '''
    def __init__(self, anchors, stride, mask, device):
        super(yolo, self).__init__()
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.anchors = self.FloatTensor(anchors)
        self.ori_anchors = self.FloatTensor(anchors)
        self.stride = stride
        self.mask = mask
        self.device = device

        self.sigmoid = nn.Sigmoid()
        # print(self.anchors, self.mask, self.stride)
        self.anchors_g = self.anchors[self.mask]/self.stride

    def forward(self, prediction):
        batch_size, channel, width, height = prediction.size() #batch_size, (5+num_classes)*3, width, height

        #prediction [2, 75, 13, 13]
        prediction = prediction.view((batch_size, width, height, 3, -1)) #[2, 13, 13, 3, 25]
        #x、y偏移量，w、h缩放值，confp置信度，classesp分类
        cxp = self.sigmoid(prediction[:, :, :, :, 0]).to(self.device)     #[2, 13, 13, 3]
        cyp = self.sigmoid(prediction[:, :, :, :, 1]).to(self.device)      #[2, 13, 13, 3]
        wp = prediction[:, :, :, :, 2].to(self.device)     #[2, 3, 13, 13]
        hp = prediction[:, :, :, :, 3].to(self.device)     #[2, 3, 13, 13]
        confp = self.sigmoid(prediction[:, :, :, :, 4]).to(self.device)     #[2, 13, 13, 3]
        classesp = self.sigmoid(prediction[:, :, :, :, 5:]).to(self.device)     #[2, 13, 13, 3, 20]

        #预测出来的坐标点的位置，加上偏移量即可
        mask_anchor_w = torch.reshape(self.anchors_g[:, 0], (1, 1, 1, 3))  #(1, 1, 1, 3)
        mask_anchor_h = torch.reshape(self.anchors_g[:, 1], (1, 1, 1, 3))  #(1, 1, 1, 3)
        x_coord = torch.arange(width).repeat(height, 1).to(self.device)    #[13, 13]
        y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).to(self.device)     #[13, 13]
        x_coord = torch.unsqueeze(torch.unsqueeze(x_coord, 0),3).to(self.device)    #[1, 13, 13, 1]
        y_coord = torch.unsqueeze(torch.unsqueeze(y_coord, 0),3).to(self.device)    #[1, 13, 13, 1]
        predx = ((cxp + x_coord) * self.stride).to(self.device)    #放到原图尺寸416x416  #[2, 13, 13, 3]
        predy = ((cyp + y_coord) * self.stride).to(self.device)     #[2, 13, 13, 3]
        predw = ((torch.exp(wp) * mask_anchor_w) * self.stride).to(self.device)      #[2, 13, 13, 3]
        predh = ((torch.exp(hp) * mask_anchor_h) * self.stride).to(self.device)     #[2, 13, 13, 3]
        # print(prediction.size(), predx.size())
        return predx, predy, predw, predh, confp, classesp, \
               cxp, cyp, wp, hp, prediction[:, :, :, :, 4], prediction[:, :, :, :, 5:]

class lossyolo(nn.Module):
    def __init__(self, iouthresh, ignore_threshold, stride, device, inputwidth):
        super(lossyolo, self).__init__()
        self.stride = stride
        self.ignore_threshold = ignore_threshold
        self.iouthresh = iouthresh
        self.device = device
        self.inputwidth = inputwidth
        self.BCElog = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
        self.MSE = nn.MSELoss(reduction='none').to(self.device)

    def forward(self, label_sbbox, label_mbbox, sbbox, mbbox, \
                           small_pre, middle_pre):
        # L_loss, L_loss_giou, L_loss_conf, L_loss_cls, Lrecall50, Lrecall75, Lnum_ins, Lobj, Lnoobj = \
        #     self.__cal_per_layer_loss(label_lbbox, lbbox, large_pre, self.stride[2])
        M_loss, M_loss_giou, M_loss_conf, M_loss_cls, Mrecall50, Mrecall75, Mnum_ins, Mobj, Mnoobj = \
            self.__cal_per_layer_loss(label_mbbox, mbbox, middle_pre, self.stride[1])
        S_loss, S_loss_giou, S_loss_conf, S_loss_cls, Srecall50, Srecall75, Snum_ins, Sobj, Snoobj = \
            self.__cal_per_layer_loss(label_sbbox, sbbox, small_pre, self.stride[0])

        loss = M_loss + S_loss
        loss_giou = M_loss_giou + S_loss_giou
        loss_conf = M_loss_conf + S_loss_conf
        loss_cls  = M_loss_cls  + S_loss_cls

        ins = (Mnum_ins + Snum_ins)
        recall50  = (Mrecall50 + Srecall50)/ins
        recall75  = (Mrecall75 + Srecall75)/ins
        obj = (Mobj + Sobj)/ins
        noobj = (Mnoobj + Snoobj)/3

        return loss, loss_giou.item(), loss_conf.item(), loss_cls.item(), \
            recall50.item(), recall75.item(), obj.item(), noobj
        
    def __cal_per_layer_loss(self, label_bbox, bbox, predict, stride):
        label_bbox = label_bbox.to(self.device)
        bbox = bbox.to(self.device)
        predx, predy, predw, predh, confp, classesp, \
                    cxp, cyp, wp, hp, confp_ori, classesp_ori = predict
        batch_size, grid = predx.size()[0], predx.size()[2]
        imgsize = grid * stride
        assert self.inputwidth == imgsize

        predxywh = torch.stack([predx, predy, predw, predh], dim = -1)
        # print(999, torch.sum(predx).item(), torch.sum(predy).item(), torch.sum(predw).item(), torch.sum(predh).item())
        label_xywh = label_bbox[..., :4]
        label_obj_mask = label_bbox[..., 4]
        label_cls  = label_bbox[..., 5:]

        giou = GIOU_xywh_torch(predxywh, label_xywh)
        # if torch.isnan(torch.sum(giou)).item()==True:
        #     print("torch.isnan(torch.sum(giou)).item()==True:")
        #     giou[...] = 0
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2] * label_xywh[..., 3] / (imgsize ** 2)
        lossgiou = label_obj_mask * bbox_loss_scale * (1.0 - giou)
        lossgiou = lossgiou.unsqueeze(-1)

        iou = iou_xywh_torch(predxywh.unsqueeze(4), bbox.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        ioumax = iou.max(-1, keepdim=True)[0].squeeze(-1)
        label_noobj_mask = (1 - label_obj_mask) * (ioumax < self.ignore_threshold).float()
        label_noobj_mask.to(self.device)

        midconf = self.BCElog(confp_ori, label_obj_mask) * (torch.pow(torch.abs(label_obj_mask - torch.sigmoid(confp_ori)), 2))
        lossconf = label_obj_mask * midconf + label_noobj_mask * midconf
        # print(classesp_ori.size(), label_cls.size())
        losscls = label_obj_mask.unsqueeze(-1) * self.BCElog(classesp_ori, label_cls)
        # print(99999999999, torch.sum(label_cls).item(), torch.sum(classesp_ori).item(), torch.sum(label_obj_mask).item())
        # print(66666666666, torch.sum(confp_ori).item(), torch.sum(midconf).item(), torch.sum(lossconf).item(), torch.sum(losscls).item())
        loss_giou = torch.sum(lossgiou)/batch_size
        loss_conf = torch.sum(lossconf)/batch_size
        loss_cls  = torch.sum(losscls)/batch_size
        # print(loss_giou, torch.sum(label_obj_mask).item(), torch.sum(bbox_loss_scale).item(), torch.sum(giou).item(), loss_conf.item(), loss_cls.item(), torch.sum(predxywh).item(),\
        # torch.sum(label_xywh).item(), imgsize)

        loss = loss_giou + loss_conf + loss_cls

        Recall = label_obj_mask * ioumax
        recall50 = torch.sum(Recall > 0.5)
        recall75 = torch.sum(Recall > 0.75)
        num_ins = torch.sum(label_obj_mask)
        
        noobj = torch.mean(label_noobj_mask * confp).item()
        if(torch.sum(label_obj_mask).item()==0):
            obj = 0
        else:
            obj = torch.sum(label_obj_mask * confp).item()
        return loss, loss_giou, loss_conf, loss_cls, recall50, recall75, num_ins, obj, noobj

class ConvBlock_LN(nn.Module):
    def __init__(self, inc, ouc, kernel_size, stride, padding, groups = 1, bias = False, rel=True):
        super(ConvBlock_LN, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inc, ouc, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(ouc)
        self.rel = rel
    def forward(self, x):
        if self.rel:
            x = self.relu(self.bn(self.conv(x)))
        else:
            x = self.bn(self.conv(x))
        return x

class yolofastest_backbone(nn.Module):
    def __init__(self, classes):
        super(yolofastest_backbone, self).__init__()
        self.block1 = ConvBlock_LN(3, 16, kernel_size = 3, stride = 2, padding = 1, groups = 1, bias = False)
        self.block2 = ConvBlock_LN(16, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block3 = ConvBlock_LN(16, 16, kernel_size = 3, stride = 1, padding = 1, groups = 16, bias = False)
        self.block4 = ConvBlock_LN(16, 8, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block5 = ConvBlock_LN(8, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block6 = ConvBlock_LN(16, 16, kernel_size = 3, stride = 1, padding = 1, groups = 16, bias = False)
        self.block7 = ConvBlock_LN(16, 8, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block8 = ConvBlock_LN(8, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block9 = ConvBlock_LN(48, 48, kernel_size = 3, stride = 2, padding = 1, groups = 48, bias = False)
        self.block10 = ConvBlock_LN(48, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block11 = ConvBlock_LN(16, 64, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block12 = ConvBlock_LN(64, 64, kernel_size = 3, stride = 1, padding = 1, groups = 64, bias = False)
        self.block13 = ConvBlock_LN(64, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block14 = ConvBlock_LN(16, 64, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block15 = ConvBlock_LN(64, 64, kernel_size = 3, stride = 1, padding = 1, groups = 64, bias = False)
        self.block16 = ConvBlock_LN(64, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block17 = ConvBlock_LN(16, 64, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block18 = ConvBlock_LN(64, 64, kernel_size = 3, stride = 2, padding = 1, groups = 64, bias = False)
        self.block19 = ConvBlock_LN(64, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block20 = ConvBlock_LN(16, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block21 = ConvBlock_LN(96, 96, kernel_size = 3, stride = 1, padding = 1, groups = 96, bias = False)
        self.block22 = ConvBlock_LN(96, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block23 = ConvBlock_LN(16, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block24 = ConvBlock_LN(96, 96, kernel_size = 3, stride = 1, padding = 1, groups = 96, bias = False)
        self.block25 = ConvBlock_LN(96, 16, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block26 = ConvBlock_LN(16, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block27 = ConvBlock_LN(96, 96, kernel_size = 3, stride = 1, padding = 1, groups = 96, bias = False)
        self.block28 = ConvBlock_LN(96, 32, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block29 = ConvBlock_LN(32, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block30 = ConvBlock_LN(192, 192, kernel_size = 3, stride = 1, padding = 1, groups = 192, bias = False)
        self.block31 = ConvBlock_LN(192, 32, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block32 = ConvBlock_LN(32, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block33 = ConvBlock_LN(192, 192, kernel_size = 3, stride = 1, padding = 1, groups = 192, bias = False)
        self.block34 = ConvBlock_LN(192, 32, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block35 = ConvBlock_LN(32, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block36 = ConvBlock_LN(192, 192, kernel_size = 3, stride = 1, padding = 1, groups = 192, bias = False)
        self.block37 = ConvBlock_LN(192, 32, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block38 = ConvBlock_LN(32, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block39 = ConvBlock_LN(192, 192, kernel_size = 3, stride = 1, padding = 1, groups = 192, bias = False)
        self.block40 = ConvBlock_LN(192, 32, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block41 = ConvBlock_LN(32, 192, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block42 = ConvBlock_LN(192, 192, kernel_size = 3, stride = 2, padding = 1, groups = 192, bias = False)
        self.block43 = ConvBlock_LN(192, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block44 = ConvBlock_LN(48, 272, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block45 = ConvBlock_LN(272, 272, kernel_size = 3, stride = 1, padding = 1, groups = 272, bias = False)
        self.block46 = ConvBlock_LN(272, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block47 = ConvBlock_LN(48, 272, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block48 = ConvBlock_LN(272, 272, kernel_size = 3, stride = 1, padding = 1, groups = 272, bias = False)
        self.block49 = ConvBlock_LN(272, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block50 = ConvBlock_LN(48, 272, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block51 = ConvBlock_LN(272, 272, kernel_size = 3, stride = 1, padding = 1, groups = 272, bias = False)
        self.block52 = ConvBlock_LN(272, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block53 = ConvBlock_LN(48, 272, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block54 = ConvBlock_LN(272, 272, kernel_size = 3, stride = 1, padding = 1, groups = 272, bias = False)
        self.block55 = ConvBlock_LN(272, 48, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block56 = ConvBlock_LN(48, 272, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block57 = ConvBlock_LN(272, 272, kernel_size = 3, stride = 2, padding = 1, groups = 272, bias = False)
        self.block58 = ConvBlock_LN(272, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block59 = ConvBlock_LN(96, 448, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block60 = ConvBlock_LN(448, 448, kernel_size = 3, stride = 1, padding = 1, groups = 448, bias = False)
        self.block61 = ConvBlock_LN(448, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block62 = ConvBlock_LN(96, 448, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block63 = ConvBlock_LN(448, 448, kernel_size = 3, stride = 1, padding = 1, groups = 448, bias = False)
        self.block64 = ConvBlock_LN(448, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block65 = ConvBlock_LN(96, 448, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block66 = ConvBlock_LN(448, 448, kernel_size = 3, stride = 1, padding = 1, groups = 448, bias = False)
        self.block67 = ConvBlock_LN(448, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block68 = ConvBlock_LN(96, 448, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block69 = ConvBlock_LN(448, 448, kernel_size = 3, stride = 1, padding = 1, groups = 448, bias = False)
        self.block70 = ConvBlock_LN(448, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block71 = ConvBlock_LN(96, 448, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block72 = ConvBlock_LN(448, 448, kernel_size = 3, stride = 1, padding = 1, groups = 448, bias = False)
        self.block73 = ConvBlock_LN(448, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block74 = ConvBlock_LN(96, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block75 = ConvBlock_LN(96, 96, kernel_size = 5, stride = 1, padding = 2, groups = 96, bias = False)
        self.block76 = ConvBlock_LN(96, 128, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block77 = ConvBlock_LN(128, 128, kernel_size = 5, stride = 1, padding = 2, groups = 128, bias = False)
        self.block78 = ConvBlock_LN(128, 128, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block79 = nn.Conv2d(128, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)

        self.block80 = ConvBlock_LN(368, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block81 = ConvBlock_LN(96, 96, kernel_size = 5, stride = 1, padding = 2, groups = 96, bias = False)
        self.block82 = ConvBlock_LN(96, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block83 = ConvBlock_LN(96, 96, kernel_size = 5, stride = 1, padding = 2, groups = 96, bias = False)
        self.block84 = ConvBlock_LN(96, 96, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False, rel = False)
        self.block85 = nn.Conv2d(96, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)

        self.dropout = nn.Dropout(p=0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = x + self.dropout(self.block7(self.block6(self.block5(x))))
        x = self.block10(self.block9(self.block8(x)))
        x = x + self.dropout(self.block13(self.block12(self.block11(x))))
        x = x + self.dropout(self.block16(self.block15(self.block14(x))))
        x = self.block19(self.block18(self.block17(x)))
        x = x + self.dropout(self.block22(self.block21(self.block20(x))))
        x = x + self.dropout(self.block25(self.block24(self.block23(x))))
        x = self.block28(self.block27(self.block26(x)))
        x = x + self.dropout(self.block31(self.block30(self.block29(x))))
        x = x + self.dropout(self.block34(self.block33(self.block32(x))))
        x = x + self.dropout(self.block37(self.block36(self.block35(x))))
        x = x + self.dropout(self.block40(self.block39(self.block38(x))))
        x = self.block43(self.block42(self.block41(x)))
        x = x + self.dropout(self.block46(self.block45(self.block44(x))))
        x = x + self.dropout(self.block49(self.block48(self.block47(x))))
        x = x + self.dropout(self.block52(self.block51(self.block50(x))))
        x = x + self.dropout(self.block55(self.block54(self.block53(x))))
        y1 = self.block56(x)
        x = self.block58(self.block57(y1))
        x = x + self.dropout(self.block61(self.block60(self.block59(x))))
        x = x + self.dropout(self.block64(self.block63(self.block62(x))))
        x = x + self.dropout(self.block67(self.block66(self.block65(x))))
        x = x + self.dropout(self.block70(self.block69(self.block68(x))))
        x = x + self.dropout(self.block73(self.block72(self.block71(x))))
        y2 = self.block74(x)
        out1 = self.block79(self.block78(self.block77(self.block76(self.block75(y2)))))
        # print(out1.size())

        x = self.upsample(y2)
        x = torch.concat([x, y1], dim = 1)
        # print(x.size())
        out2 = self.block85(self.block84(self.block83(self.block82(self.block81(self.block80(x))))))
        return out1, out2 #out2 small obj   out1 big obj

class yolofastestNet(nn.Module):
    def __init__(self, num_classes, anchors, strides, ignore_thresh, inputwidth,device,\
        score_thresh = 0.45, nms_thresh = 0.45):
        super(yolofastestNet, self).__init__()
        self.num_classes = num_classes
        self.net = inputnet()
        self.anchors = anchors
        self.strides = strides
        self.ignore_thresh = ignore_thresh
        self.score_thresh = score_thresh
        print('self.score_thresh: ', self.score_thresh)
        print('nms_thresh: ', nms_thresh)
        time.sleep(3)
        self.nms_thresh = nms_thresh
        self.inputwidth = inputwidth
        self.device = device
        self.yolo3 = yolo(self.anchors, self.strides[1], 1, self.device)
        self.yolo2 = yolo(self.anchors, self.strides[0], 0, self.device)
        # self.yolo1 = yolo(self.anchors, self.strides[0], 0, self.device)
        self.yolofastest_backbone = yolofastest_backbone(num_classes)

    def forward(self, x, predict):
        out1, out2 = self.yolofastest_backbone(x) #out2 small obj   out1 big obj
        #进入header层并上采样    大目标  特征图缩放尺度大网络深，特征图本身尺寸小 13x13
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp, cxp, cyp, wp, hp
        large_pre = self.yolo3(out1)

        #进入header层并上采样    small的目标  特征图缩放尺度中等网络中部，特征图本身尺寸中等 26x26
        #y2 [2, 255, 26, 26]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp, cxp, cyp, wp, hp
        middle_pre = self.yolo2(out2)

        #[2, 255, 13, 13]、[2, 255, 26, 26]
        if not predict:
            #训练时返回值
            return middle_pre, large_pre
        else:
            #检测或者测试时返回值
            results = [large_pre[:-6], middle_pre[:-6]]
            ##[2, 13, 13, 3]、~、~、~、~、#[2, 13, 13, 20, 3]
            batch_size = results[0][0].size()[0]
            prediction = [[]]*batch_size
            #[2, 13, 13, 3]、[2, 13, 13, 3]、[2, 13, 13, 3]、[2, 13, 13, 3]、[2, 13, 13, 3]、[2, 13, 13, 3, 20])
            for predx, predy, predw, predh, confp, classesp in results:
                feature_scale = predx.size()[2] #13、26、52

                confp = torch.unsqueeze(confp, -1)    ##[2, 13, 13, 3， 1]
                scoresp = confp*classesp             ##[2, 13, 13, 3, 20]
                scoresmaxp, scores_labelp = torch.max(scoresp, dim=4)      ##[2, 13, 13, 3]、[2, 13, 13, 3]
                mask = scoresmaxp > self.score_thresh    ##[2, 13, 13, 3]

                # scoresmaxp, scores_labelp = torch.max(classesp, dim=4)      ##[2, 13, 13, 3]、[2, 13, 13, 3]
                # mask = scoresmaxp>self.score_thresh    ##[2, 13, 13, 3]
                for bs in range(batch_size):
                    mas = mask[bs, :, :, :]
                    if torch.sum(mas).item()==0:
                        continue
                    predx = torch.unsqueeze(predx[bs, mas], 1)/self.inputwidth
                    predy = torch.unsqueeze(predy[bs,mas], 1)/self.inputwidth
                    predw = torch.unsqueeze(predw[bs,mas], 1)/self.inputwidth
                    predh = torch.unsqueeze(predh[bs,mas], 1)/self.inputwidth
                    scoresmaxp = torch.unsqueeze(scoresmaxp[bs, mas], 1) 
                    scores_labelp = torch.unsqueeze(scores_labelp[bs, mas], 1)
                    predict = list(torch.cat([predx, predy, predw, predh, scoresmaxp, scores_labelp], dim=1).detach().cpu().numpy())
                    prediction[bs].extend(predict)
            for bs in range(batch_size):
                prediction[bs] = nms(np.array(prediction[bs]), self.nms_thresh)
            prediction = torch.tensor(prediction)
            return prediction, torch.zeros(1), torch.zeros(1)

def getdata():
    # image = torch.rand((2, 3, 416,416))
    image1 = cv2.imread(r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\JPEGImages\000547.jpg')
    image2 = cv2.imread(r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\JPEGImages\009854.jpg')
    image1 = cv2.resize(image1, (416, 416))
    image2 = cv2.resize(image2, (416, 416))
    image = np.stack([image1, image2])
    image = np.transpose(image, (0, 3, 1, 2))# batch_size, channel, width, height
    image = torch.Tensor(image)

    gtpath1 = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\labels\000547.txt'
    gtpath2 = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\labels\009854.txt'
    gt1 = []
    with open(gtpath1, 'r') as f:
        for i in f.readlines():
            label, cx, cy, w, h = i.strip().split(' ')
            gt1.append([int(label), float(cx), float(cy), float(w), float(h), 0])
    with open(gtpath2, 'r') as f:
        for i in f.readlines():
            label, cx, cy, w, h = i.strip().split(' ')
            gt1.append([int(label), float(cx), float(cy), float(w), float(h), 1])
    gt1 = np.array(gt1)
    return image, ''#gt1

if __name__ == '__main__':
    num_classes = 20 #voc2007存在20类
    inputwidth = 416
    anchors = [[[12,22], [36,58], [71,137]], \
                [[126,271], [215,150], [305,350]]]
    ignore_thresh = 0.7 #iou小于0.7的看作负样本，只计算confidence的loss
    score_thresh = 0.45
    nms_thresh = 0.35
    image = torch.randn((1, 3, 416, 416))
    gt = [[0, 0.2, 0.2, 0.1, 0.1], \
          [0, 0.2, 0.2, 0.1, 0.1]]
    strides = [0, 1, 2]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolofastest = yolofastest_prune37(num_classes, anchors, strides, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
    out1, out2 = yolofastest(image, gt)
    print(result3, result2, result1, precision50, recall50, recall75)
