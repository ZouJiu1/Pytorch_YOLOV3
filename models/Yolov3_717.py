#Author：ZouJiu
#Time: 2021-8-13

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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

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

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
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
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
    return GIOU

class yolo(nn.Module):
    '''
    对输出的特征图进行解码，得到坐标、置信度、分类的概率
    并计算相应的objectness loss、classify loss、coordinates loss
    '''
    def __init__(self, num_classes, anchors, strides, ignore_thresh, mask,device,\
         inputwidth, score_thresh = 0.45, nms_thresh = 0.35):
        super(yolo, self).__init__()
        self.num_classes = num_classes
        self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
        self.BoolTensor = torch.cuda.BoolTensor if torch.cuda.is_available() else torch.BoolTensor
        self.anchors = self.FloatTensor(anchors)
        self.ori_anchors = self.FloatTensor(anchors)
        self.strides = self.FloatTensor(strides)
        self.ignore_thresh = ignore_thresh
        self.mask = mask
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.device = device

        #inputwidth 是32的正整数倍
        self.inputwidth = inputwidth
        self.ori_a_w = self.ori_anchors[:, 0]/self.inputwidth  #相对于原图的归一化的尺寸
        self.ori_a_h = self.ori_anchors[:, 1]/self.inputwidth   #相对于原图的归一化的尺寸
        self.areas      = self.inputwidth ** 2
        self.sigmoid = nn.Sigmoid()
        self.anchors_g = self.anchors[self.mask]
        self.BCE = nn.BCELoss(reduction='none').to(self.device)
        self.MSE = nn.MSELoss(reduction='none').to(self.device)
        # self.CE = nn.CrossEntropyLoss().to(self.device)
        self.strides = torch.unsqueeze(self.strides, 1)
        self.anchors = self.anchors/self.strides             #[9, 2]
        self.scale_noobj = 0.5
        self.scale_obj   = 5
        # self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(self.device)
        # self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(self.device)

    def forward(self, prediction, gt):
        self.gt = gt
        batch_size, channel, width, height = prediction.size() #batch_size, (5+num_classes)*3, width, height

        #prediction [2, 75, 13, 13]
        prediction = prediction.view((batch_size, 3, width, height, -1)) #[2, 3, 13, 13, 25]
        #x、y偏移量，w、h缩放值，confp置信度，classesp分类
        cxp = self.sigmoid(prediction[:, :, :, :, 0]).to(self.device)     #[2, 3, 13, 13]
        cyp = self.sigmoid(prediction[:, :, :, :, 1]).to(self.device)      #[2, 3, 13, 13]
        wp = prediction[:, :, :, :, 2].to(self.device)     #[2, 3, 13, 13]
        hp = prediction[:, :, :, :, 3].to(self.device)     #[2, 3, 13, 13]
        confp = self.sigmoid(prediction[:, :, :, :, 4]).to(self.device)     #[2, 3, 13, 13]
        classesp = self.sigmoid(prediction[:, :, :, :, 5:]).to(self.device)     #[2, 3, 13, 13, 20]

        #预测出来的坐标点的位置，加上偏移量即可
        mask_anchor_w = torch.reshape(self.anchors_g[:, 0], (1, 3, 1, 1))  #(1, 3, 1, 1)
        mask_anchor_h = torch.reshape(self.anchors_g[:, 1], (1, 3, 1, 1))  #(1, 3, 1, 1)
        x_coord = torch.arange(width).repeat(height, 1).to(self.device)    #[13, 13]
        y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).to(self.device)     #[13, 13]
        x_coord = torch.unsqueeze(torch.unsqueeze(x_coord, 0),0).to(self.device)    #[1, 1, 13, 13]
        y_coord = torch.unsqueeze(torch.unsqueeze(y_coord, 0),0).to(self.device)    #[1, 1, 13, 13]
        predx = cxp + x_coord      #[2, 3, 13, 13]
        predy = cyp + y_coord     #[2, 3, 13, 13]
        predw = ((torch.exp(wp) * mask_anchor_w) / self.inputwidth) * width      #[2, 3, 13, 13]
        predh = ((torch.exp(hp) * mask_anchor_h) / self.inputwidth) * height     #[2, 3, 13, 13]

        if len(self.gt)==0:
            return predx, predy, predw, predh, confp, classesp, 3, 3, 3

        numpic = self.gt[:, -1]
        assert batch_size==numpic[-1] + 1

        num_anchors = 3

        gtctx = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        gtcty = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        gtw = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        gth = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)

        offsetctx = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        offsetcty = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        offsetw = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        offseth = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        objectness = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        smallscales = self.FloatTensor(batch_size, num_anchors, height, width).fill_(0)
        classes = self.FloatTensor(batch_size, num_anchors, height, width, self.num_classes).fill_(0)
        objmask = torch.BoolTensor(batch_size, num_anchors, height, width).fill_(False).to(self.device)
        noobjconfmask = torch.BoolTensor(batch_size, num_anchors, height, width).fill_(True).to(self.device)

        #对于每个label计算和anchor之间IOU最匹配的特征层，不考虑中心点位置
        lossiou = 0
        glossiou = []

        #将真实label的框放到特征图里
        iou_col = torch.zeros(0).to(self.device)
        ioulos_col = torch.zeros(0).to(self.device)
        num_correct = 0
        all_gtnums  = len(self.gt)

        for bs in range(batch_size):
            batch_gt = self.gt[self.gt[:, -1]==bs][:, :-1]
            gtxyxy = torch.zeros((batch_gt.size()[0], 3 + 1), dtype=torch.float32).to(self.device) #[gt, 4]
            gtxyxy[:, 0] = (batch_gt[:, 1] - batch_gt[:, 3] * 0.5) * self.inputwidth
            gtxyxy[:, 1] = (batch_gt[:, 2] - batch_gt[:, 4] * 0.5) * self.inputwidth
            gtxyxy[:, 2] = (batch_gt[:, 1] + batch_gt[:, 3] * 0.5) * self.inputwidth
            gtxyxy[:, 3] = (batch_gt[:, 2] + batch_gt[:, 4] * 0.5) * self.inputwidth

            '''
            a_n  = 3    #3 anchors
            prxyxy = torch.zeros((a_n, 3 + 1), dtype=torch.float32).to(self.device)   #[a_n, 4]
            for i in range(height):
                for j in range(width):
                    prxyxy[:, 0]  = ((predx[bs, :, i, j] - predw[bs, :, i, j] * 0.5)/width) * self.inputwidth
                    prxyxy[:, 1]  = ((predy[bs, :, i, j] - predh[bs, :, i, j] * 0.5)/height) * self.inputwidth
                    prxyxy[:, 2]  = ((predx[bs, :, i, j] + predw[bs, :, i, j] * 0.5)/width) * self.inputwidth
                    prxyxy[:, 3]  = ((predy[bs, :, i, j] + predh[bs, :, i, j] * 0.5)/height) * self.inputwidth
                    g_p_iou = box_iou(gtxyxy, prxyxy)   #[gt, a_n]
                    max_val, max_ind  = torch.max(g_p_iou, dim = 0) #[a_n]
                    for mv in range(len(max_val)):
                        if max_val[mv] > self.ignore_thresh:
                            noobjconfmask[bs, mv, i, j] = False
            '''
            a_n  = 3    #3 anchors
            prxyxy = torch.zeros((a_n, width, height, 3 + 1), dtype=torch.float32).to(self.device)   #[a_n, 4]
            prxyxy[:, :, :, 0]  = ((predx[bs, :, :, :] - predw[bs, :, :, :] * 0.5)/width) * self.inputwidth   #[num_anchors, height, width, 4]
            prxyxy[:, :, :, 1]  = ((predy[bs, :, :, :] - predh[bs, :, :, :] * 0.5)/height) * self.inputwidth  #[num_anchors, height, width, 4]
            prxyxy[:, :, :, 2]  = ((predx[bs, :, :, :] + predw[bs, :, :, :] * 0.5)/width) * self.inputwidth   #[num_anchors, height, width, 4]
            prxyxy[:, :, :, 3]  = ((predy[bs, :, :, :] + predh[bs, :, :, :] * 0.5)/height) * self.inputwidth  #[num_anchors, height, width, 4]
            prxyxy = prxyxy.view((-1, 4))   #[num_anchors*height*width, 4]
            g_p_iou = box_iou(gtxyxy, prxyxy)   #[gt, num_anchors*height*width]
            g_p_iou = g_p_iou.view((g_p_iou.size()[0], num_anchors, height, width))
            max_val, max_ind  = torch.max(g_p_iou, dim = 0)  #[num_anchors, height, width] max value for every masked anchor
            gt_allpreds_iou = max_val > self.ignore_thresh
            noobjconfmask[bs, gt_allpreds_iou] = False
            
            for ind, rectangle in enumerate(batch_gt):
                label = int(rectangle[0])
                i = int(rectangle[1] * width) #feature_map center x
                j = int(rectangle[2] * height) #feature_map center y
                gtone = torch.zeros((1, 3 + 1), dtype=torch.float32).to(self.device)    #xmin  ymin  xmax  ymax
                gtone[:, 0] = (rectangle[1] - rectangle[3] * 0.5) * self.inputwidth
                gtone[:, 1] = (rectangle[2] - rectangle[4] * 0.5) * self.inputwidth
                gtone[:, 2] = (rectangle[1] + rectangle[3] * 0.5) * self.inputwidth
                gtone[:, 3] = (rectangle[2] + rectangle[4] * 0.5) * self.inputwidth

                a_total = 9 #total 9 anchors
                anchorxyxy = torch.zeros((a_total, 3+1)).to(self.device)
                anchorxyxy[:, 0] = (rectangle[1] - self.ori_a_w * 0.5) * self.inputwidth
                anchorxyxy[:, 1] = (rectangle[2] - self.ori_a_h * 0.5) * self.inputwidth
                anchorxyxy[:, 2] = (rectangle[1] + self.ori_a_w * 0.5) * self.inputwidth
                anchorxyxy[:, 3] = (rectangle[2] + self.ori_a_h * 0.5) * self.inputwidth
                g_a_iou = box_iou(gtxyxy, anchorxyxy)[0] #[1, 9]
                g_a_max, g_a_maxind  = torch.max(g_a_iou, dim = 0)
                if(g_a_maxind.item() in self.mask): #属于该层的yolo_layer predict
                    a_ind = self.mask.index(g_a_maxind.item())

                    objmask[bs, a_ind, i, j] = True
                    noobjconfmask[bs, a_ind, i, j] = False
                    predone = torch.zeros((1, 3 + 1), dtype=torch.float32).to(self.device) #[1, 4]
                    predone[0, 0]  = ((predx[bs, a_ind, i, j] - predw[bs, a_ind, i, j] * 0.5)/width) * self.inputwidth
                    predone[0, 1]  = ((predy[bs, a_ind, i, j] - predh[bs, a_ind, i, j] * 0.5)/width) * self.inputwidth
                    predone[0, 2]  = ((predx[bs, a_ind, i, j] + predw[bs, a_ind, i, j] * 0.5)/width) * self.inputwidth
                    predone[0, 3]  = ((predy[bs, a_ind, i, j] + predh[bs, a_ind, i, j] * 0.5)/width) * self.inputwidth
                    one_iou = box_iou(gtone, predone)[0]   #[1]

                    objectness[bs, a_ind, i, j] = 1
                    classes[bs, a_ind, i, j, label] = 1
                    
                    gtwidth  = rectangle[3] * self.inputwidth
                    gtheight = rectangle[4] * self.inputwidth
                    offsetctx[bs, a_ind, i, j] = rectangle[1] * width - i
                    offsetcty[bs, a_ind, i, j] = rectangle[2] * height - j
                    ofw = torch.log(gtwidth/self.ori_anchors[self.mask[a_ind], 0])
                    ofh = torch.log(gtheight/self.ori_anchors[self.mask[a_ind], 1])

                    offsetw[bs, a_ind, i, j] = ofw.float()
                    offseth[bs, a_ind, i, j] = ofh.float()

                    # boxloss_scale =   2.0 - ((1.0 * (gtwidth * scale) * (gtheight * scale)) / self.areas)
                    boxloss_scale = 2.0 - rectangle[3] * rectangle[4]
                    smallscales[bs, a_ind, i, j] = boxloss_scale

                    iou_col       =   torch.cat([iou_col, one_iou])

                    iou_los       =   (1 - one_iou) * boxloss_scale
                    ioulos_col    =   torch.cat([ioulos_col, iou_los])
                    
                    predlabel = torch.argmax(classesp[bs, a_ind, i, j]).item()
                    score = confp[bs, a_ind, i, j].item()
                    if ( one_iou.item() > 0.5 and predlabel==label and score > 0.38):
                        num_correct += 1
        
        nProposals = int((confp > 0.5).sum().item())
        # recall = float(num_correct/all_gtnums) if all_gtnums else 1
        # precision = float(num_correct/(nProposals+1e-10))

        offsetctx = Variable(offsetctx.type(torch.FloatTensor), requires_grad=False).to(self.device)
        offsetcty = Variable(offsetcty.type(torch.FloatTensor), requires_grad=False).to(self.device)
        offsetw = Variable(offsetw.type(torch.FloatTensor), requires_grad=False).to(self.device)
        offseth = Variable(offseth.type(torch.FloatTensor), requires_grad=False).to(self.device)
        objectness = Variable(objectness.type(torch.FloatTensor), requires_grad=False).to(self.device)
        classes = Variable(classes.type(torch.FloatTensor), requires_grad=False).to(self.device)
        smallscales = Variable(smallscales.type(torch.FloatTensor), requires_grad=False).to(self.device)
        clsmask = classes.type(torch.BoolTensor).to(self.device)
        
        # conf_mask_true = objmask
        # conf_mask_false = noobjconfmask # - objmask
        # print(torch.sum(objmask), torch.sum(noobjconfmask), 13*13*3*9 - torch.sum(objmask|noobjconfmask))

        objmask = Variable(objmask.type(torch.BoolTensor), requires_grad=False).to(self.device)
        # conf_mask_true = Variable(conf_mask_true.type(torch.BoolTensor), requires_grad=False).to(self.device)
        noobjconfmask = Variable(noobjconfmask.type(torch.BoolTensor), requires_grad=False).to(self.device)

        # self.BCE_scale = nn.BCELoss(weight=smallscales[objmask]).to(self.device)
        # wp, offsetw = wp * smallscales, offsetw * smallscales
        # hp, offseth = hp * smallscales, offseth * smallscales
        # self.BCELoss = nn.BCELoss().to(self.device)
        # print(cxp[objmask].size(), smallscales[objmask].size())
        # lossx         =  self.MSE(cxp[objmask] * smallscales[objmask], offsetctx[objmask] * smallscales[objmask])
        # lossy         =  self.MSE(cyp[objmask] * smallscales[objmask], offsetcty[objmask] * smallscales[objmask])
        lossx         =  smallscales[objmask] * self.BCE(cxp[objmask], offsetctx[objmask]) #0.7 larger than MSE 0.2 
        lossy         =  smallscales[objmask] * self.BCE(cyp[objmask], offsetcty[objmask]) #larger than MSE
        # print(wp[objmask], offsetw[objmask], hp[objmask], offseth[objmask], smallscales[objmask])
        lossw         =  0.5 * smallscales[objmask] * self.MSE(wp[objmask] * smallscales[objmask], offsetw[objmask] * smallscales[objmask])
        lossh         =  0.5 * smallscales[objmask] * self.MSE(hp[objmask] * smallscales[objmask], offseth[objmask] * smallscales[objmask])

        lossx = torch.sum(lossx)/batch_size
        lossy = torch.sum(lossy)/batch_size
        lossw = torch.sum(lossw)/batch_size
        lossh = torch.sum(lossh)/batch_size
        # lossx         =  self.MSE(cxp[objmask], offsetctx[objmask])
        # lossy         =  self.MSE(cyp[objmask], offsetcty[objmask])
        # lossw         =  self.MSE(wp[objmask], offsetw[objmask])
        # lossh         =  self.MSE(hp[objmask], offseth[objmask])

        self.scale_noobj = 1
        self.scale_obj   = 1

        # print("confidence: ", confp[conf_mask_true], objectness[conf_mask_true], torch.sum(objectness[conf_mask_true]), torch.sum(objectness[conf_mask_false]))
        # print(torch.sum(confp[conf_mask_true]), torch.sum(confp[conf_mask_false]))
        # print('IOU: ', torch.sum(iou_col > 0.5), torch.sum(classes))
        # print(objectness[conf_mask_false].size(), objectness[conf_mask_true].size(), classes[objmask].size())
        
        loss = 0 #(lossx + lossy + lossw + lossh) + lossobj + lossclasses # + lossiou

        # lossnoobj = torch.mean(torch.log(1 - confp[conf_mask_false]) * -1)  #noobjectness
        # lossobj = torch.mean(torch.log(confp[conf_mask_true]) * -1)
        # # print(torch.log(confp[conf_mask_true]), torch.log(confp[conf_mask_true]) * -1, torch.mean(torch.log(confp[conf_mask_true]) * -1))
        # gt = classes[objmask]
        # pd = classesp[objmask]
        # tmp = torch.tensor(0, dtype=torch.float32)
        # for i in range(pd.size()[0]):
        #     for j in range(self.num_classes):
        #         tmp += -gt[i][j] * torch.log(pd[i][j]) - (1 - gt[i][j]) * torch.log(1- pd[i][j])
        # if (pd.size()[0] * self.num_classes)!=0:
        #     lossclasses = tmp/(pd.size()[0] * self.num_classes)
        # else:
        #     lossclasses = tmp

        # self.BCELOGITS = nn.BCEWithLogitsLoss().to(self.device)
        # lossnoobj   =  self.scale_noobj *  self.BCELOGITS(prediction[:, :, :, :, 4][conf_mask_false], objectness[conf_mask_false])
        # lossobj     =  self.scale_obj * self.BCELOGITS(prediction[:, :, :, :, 4][conf_mask_true], objectness[conf_mask_true])    #self.BCE(confp, objectness)
        # lossclasses   =  self.BCELOGITS(prediction[:, :, :, :, 5:][objmask], classes[objmask])
        # print(classesp[objmask], classes[objmask])

        # lossnoobj   =  torch.abs(confp[noobjconfmask] - objectness[noobjconfmask]) #(1 - objectness) * self.BCE(confp, objectness) * self.MSE(confp, objectness) * noobjconfmask
        # lossobj     =  torch.abs(confp[objmask] - objectness[objmask])   #objectness * self.BCE(confp, objectness) * self.MSE(confp, objectness)    #self.BCE(confp, objectness)
        lossnoobj   =  (1 - objectness) * self.BCE(confp, objectness) * self.MSE(confp, objectness) * noobjconfmask
        lossobj     =  objectness * self.BCE(confp, objectness) * self.MSE(confp, objectness)    #self.BCE(confp, objectness)
        lossclasses =  self.BCE(classesp[objmask], classes[objmask])

        lossnoobj   =  torch.sum(lossnoobj)/batch_size
        lossobj     =  torch.sum(lossobj)/batch_size
        lossclasses =  torch.sum(lossclasses)/batch_size

        recall50 = torch.sum(iou_col > 0.5)
        recall75 = torch.sum(iou_col > 0.75)

        if(len(ioulos_col)==0):
            lossiou = torch.zeros(1).to(self.device)
        else:
            lossiou = torch.mean(ioulos_col)
        # losslis = {"lossx": lossx, "lossy":lossy, "lossw":lossw, "lossh":lossh, \
        #     "lossobj":lossobj, "lossnoobj":lossnoobj, "lossclasses":lossclasses}

        # box = 1
        # cls = 1
        # obj = 1

        loss = lossx + lossy + lossw + lossh + lossnoobj + lossclasses + lossobj
        # print(lossx , lossy , lossw , lossh , lossnoobj , lossclasses , lossobj)

        # for key, value in losslis.items():
        #     if(torch.isnan(value).item()==True):
        #         # print(key, " is nan")
        #         pass
        #     else:
        #         if 'lossx'==key or 'lossy'==key or 'lossw'==key or 'lossh'==key:
        #             loss += box * value
        #         elif 'lossclasses'==key:
        #             loss += cls * value
        #         elif 'lossobj'==key:
        #             loss += obj * value
        #         else:
        #             loss += value
        #         print('{:.3f}'.format(value.item()), end = ' ')
        # print()

        if(torch.sum(objmask).item()==0):
            objectnessscore = 0
            cls = 0
        else:
            objectnessscore = torch.sum(confp * objectness).item()
            cls = torch.sum(classesp[clsmask]).item()

        noobjectness = torch.mean(confp[noobjconfmask]).item()
        return loss, objectnessscore, recall50.item(), recall75.item(), noobjectness, lossiou, num_correct, nProposals, cls

class Yolov3(nn.Module):
    def __init__(self, num_classes, anchors, strides, ignore_thresh, inputwidth,device,\
        score_thresh = 0.45, nms_thresh = 0.45):
        super(Yolov3, self).__init__()
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
        self.ResBlock1 = resblock1()
        self.ResBlock2 = resblock2()
        self.ResBlock3 = resblock3()
        self.ResBlock4 = resblock4()
        self.ResBlock5 = resblock5()
        self.yolo3 = yolo(self.num_classes, self.anchors, self.strides, \
            self.ignore_thresh, [6, 7, 8], self.device, self.inputwidth,\
            score_thresh = self.score_thresh, nms_thresh = self.nms_thresh)
        self.yolo2 = yolo(self.num_classes, self.anchors, self.strides, \
            self.ignore_thresh, [3, 4, 5], self.device,self.inputwidth, \
                score_thresh = self.score_thresh, nms_thresh = self.nms_thresh)
        self.yolo1 = yolo(self.num_classes, self.anchors, self.strides, \
            self.ignore_thresh, [0, 1, 2], self.device,self.inputwidth, \
                score_thresh = self.score_thresh, nms_thresh = self.nms_thresh)
        self.hb3 = header_block(self.num_classes, 1024, 512, upornot=True) #r3.size()[1] 第二个参数
        self.hb2 = header_block(self.num_classes, 768, 256, upornot=True) #r2up.size()[1] 第二个参数
        self.hb1 = header_block(self.num_classes, 384, 128, upornot=False) #r1up.size()[1] 第二个参数

    def forward(self, x, gt=''):
        self.gt = gt
        r = self.net(x)
        r = self.ResBlock1(r)
        r = self.ResBlock2(r)
        upy1, r1  = self.ResBlock3(r)
        upy2, r2 = self.ResBlock4(r1)
        r3 = self.ResBlock5(r2)
        
        #进入header层并上采样
        up3, y3 = self.hb3(r3) #y3 [2, 255, 13, 13]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp
        result3, objectness3, recall50_3, recall75_3, noobjectness3, \
            lossiou3, num_correct3, nProposals3, classes3 = self.yolo3(y3, deepcopy(self.gt))

        #进入header层并上采样
        r2up = torch.cat([up3, upy2], dim=1)
        up2, y2 = self.hb2(r2up) #y2 [2, 255, 26, 26]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp
        result2, objectness2, recall50_2, recall75_2, noobjectness2, \
            lossiou2, num_correct2, nProposals2, classes2 = self.yolo2(y2, deepcopy(self.gt))

        #进入header层
        r1up = torch.cat([up2, upy1], dim=1)
        y1 = self.hb1(r1up) #y1 [2, 255, 52, 52]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp
        result1, objectness1, recall50_1, recall75_1, noobjectness1, \
            lossiou1, num_correct1, nProposals1, classes1 = self.yolo1(y1, deepcopy(self.gt))

        #[2, 255, 13, 13]、[2, 255, 26, 26]、[2, 255, 52, 52]
        if len(self.gt)!=0:
            #训练时返回值
            length = len(np.where(np.array([objectness3,objectness2,objectness1])!=0)[0])
            # print(objectness3, objectness2, objectness1, length)
            objectness = (objectness3 + objectness2 + objectness1)/len(self.gt)
            noobjectness = (noobjectness3 + noobjectness2 + noobjectness1)/3
            recall50 = (recall50_1 + recall50_2 + recall50_3)/len(self.gt)
            recall75 = (recall75_1 + recall75_2 + recall75_3)/len(self.gt)
            correct = num_correct1 + num_correct2 + num_correct3
            recall = correct/len(self.gt)
            precision = correct/(nProposals1 + nProposals2 + nProposals3 + 1e-10)
            class_score = (classes1 + classes2 + classes3)/len(self.gt)
            return result3, result2, result1, objectness, recall50, recall75, noobjectness, recall, precision, class_score
        else:
            #检测或者测试时返回值
            results = [[result3, objectness3, recall50_3, recall75_3, noobjectness3, lossiou3],\
                [result2, objectness2, recall50_2, recall75_2, noobjectness2, lossiou2],\
                [result1, objectness1, recall50_1, recall75_1, noobjectness1, lossiou1]]
            ##[2, 3, 13, 13]、~、~、~、~、#[2, 3, 13, 13, 20]
            batch_size = results[0][0].size()[0]
            prediction = [[]]*batch_size
            #[2, 3, 13, 13]、[2, 3, 13, 13]、[2, 3, 13, 13]、[2, 3, 13, 13]、[2, 3, 13, 13]、[2, 3, 13, 13, 20])
            for predx, predy, predw, predh, confp, classesp in results:
                feature_scale = predx.size()[-1] #13、26、52

                confp = torch.unsqueeze(confp, 4)    ##[2, 3, 13, 13, 1]
                scoresp = confp*classesp             ##[2, 3, 13, 13, 20]
                scoresmaxp, scores_labelp = torch.max(scoresp, dim=4)      ##[2, 3, 13, 13]、[2, 3, 13, 13]
                mask = scoresmaxp > self.score_thresh    ##[2, 3, 13, 13]

                # scoresmaxp, scores_labelp = torch.max(classesp, dim=4)      ##[2, 3, 13, 13]、[2, 3, 13, 13]
                # mask = scoresmaxp>self.score_thresh    ##[2, 3, 13, 13]
                for bs in range(batch_size):
                    mas = mask[bs, :, :, :]
                    if torch.sum(mas).item()==0:
                        continue
                    predx = torch.unsqueeze(predx[bs, mas], 1)/feature_scale
                    predy = torch.unsqueeze(predy[bs,mas], 1)/feature_scale
                    predw = torch.unsqueeze(predw[bs,mas], 1)/feature_scale
                    predh = torch.unsqueeze(predh[bs,mas], 1)/feature_scale
                    scoresmaxp = torch.unsqueeze(scoresmaxp[bs, mas], 1) 
                    scores_labelp = torch.unsqueeze(scores_labelp[bs, mas], 1)
                    predict = list(torch.cat([predx, predy, predw, predh, scoresmaxp, scores_labelp], dim=1).detach().cpu().numpy())
                    prediction[bs].extend(predict)
            for bs in range(batch_size):
                prediction[bs] = nms(np.array(prediction[bs]), self.nms_thresh)
            prediction = torch.tensor(prediction)
            return prediction, torch.zeros(1), torch.zeros(1), torch.zeros(1), \
                    torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

            #     predict = []
            #     feature_scale = predx.size()[-1] #13、26、52
            #     batch_size = predx.size()[0]
            #     for bs in range(batch_size):
            #         for an in range(3):
            #             for i in range(feature_scale):
            #                 for j in range(feature_scale):
            #                     scoresp = classesp[bs, an, :, j, i]*confp[bs, an, j, i]
            #                     labelp = torch.argmax(scoresp).item()
            #                     scoresmaxp = torch.max(scoresp).item()
            #                     if scoresmaxp>self.score_thresh:
            #                         predict.append([predx[bs,an,j,i].item(), predy[bs,an,j,i].item(), \
            #                             predw[bs,an,j,i].item(), predh[bs,an,j,i].item(),\
            #                                 scoresmaxp, labelp])
            #     predict = np.array(predict)
            #     predict = nms(predict, self.nms_thresh)
            #     prediction.extend(predict)
            #     print('after nms: ', len(predict))
            # prediction = np.array(prediction)
            # print('all: ',prediction.shape)
            # return prediction, 2,  2,  2,  2,  2
            


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
    anchors = [[10,13], [16,30], [33,23],\
        [30,61],  [62,45],  [59,119],  \
        [116,90],  [156,198],  [373,326]]
    ignore_thresh = 0.7 #iou小于0.7的看作负样本，只计算confidence的loss
    score_thresh = 0.45
    nms_thresh = 0.35
    image, gt = getdata()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # gt = ''

    yolov3 = Yolov3(num_classes, anchors, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
    result3, result2, result1, precision50, recall50, recall75 = yolov3(image, gt)
    print(result3, result2, result1, precision50, recall50, recall75)
