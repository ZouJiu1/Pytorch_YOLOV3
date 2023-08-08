#Author：ZouJiu
#Time: 2021-8-13

import os
import cv2
import time
import torch
from copy import deepcopy 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class inputnet(nn.Module):
    def __init__(self):
        super(inputnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
       x = self.leaky_relu(self.norm1(self.conv1(x)))
       x = self.leaky_relu(self.norm2(self.conv2(x)))
       return x

class resblock1(nn.Module):
    def __init__(self):
        super(resblock1, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        #self.relu = nn.ReLU()
        
        self.conv_extra = nn.Conv2d(64, 128, 3, stride=2, padding=1)
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
        self.conv1 = nn.Conv2d(128, 64, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
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
        self.conv_extra = nn.Conv2d(128, 256, 3, stride=2, padding=1)
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
        self.conv1 = nn.Conv2d(256, 128, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
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
        self.conv_extra = nn.Conv2d(256, 512, 3, stride=2, padding=1)
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
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
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
        self.conv_extra = nn.Conv2d(512, 1024, 3, stride=2, padding=1)
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
        self.conv1 = nn.Conv2d(1024, 512, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 1024, 3, stride=1, padding=1)
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
        self.conv1 = nn.Conv2d(beginchannel, channel, 1, stride=1)
        self.norm1 = nn.BatchNorm2d(channel)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(channel, channel*2, 3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(channel*2)
        self.conv3 = nn.Conv2d(channel*2, channel, 1, stride=1)
        self.norm3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, channel*2, 3, stride=1, padding=1)
        self.norm4 = nn.BatchNorm2d(channel*2)
        self.conv5 = nn.Conv2d(channel*2, channel, 1, stride=1)
        self.norm5 = nn.BatchNorm2d(channel) #准备上采样

        #yolo1,接conv5
        self.conv6 = nn.Conv2d(channel, channel*2, 3, stride=1, padding=1)
        self.norm6 = nn.BatchNorm2d(channel*2)
        self.conv7 = nn.Conv2d(channel*2, (5+num_classes)*3, 1, stride=1)
        self.norm7 = nn.BatchNorm2d((5+num_classes)*3)
        #self.relu = nn.ReLU()

        #upsample,接conv5
        self.conv8 = nn.Conv2d(channel, channel//2, 1, stride=1)
        self.norm8 = nn.BatchNorm2d(channel//2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.leaky_relu(self.norm1(self.conv1(x)))
        x = self.leaky_relu(self.norm2(self.conv2(x)))
        x = self.leaky_relu(self.norm3(self.conv3(x)))
        x = self.leaky_relu(self.norm4(self.conv4(x)))
        x = self.leaky_relu(self.norm5(self.conv5(x)))

        y3 = self.leaky_relu(self.norm6(self.conv6(x)))
        y3 = self.conv7(y3)
        
        if self.upornot:
            up3 = self.leaky_relu(self.norm8(self.conv8(x)))
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
        self.strides = self.FloatTensor(strides)
        self.ignore_thresh = ignore_thresh
        self.mask = mask
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.device = device

        #inputwidth 是32的正整数倍
        self.inputwidth = inputwidth
        self.areas      = self.inputwidth ** 2
        self.sigmoid = nn.Sigmoid()
        self.anchors_g = self.anchors[self.mask]
        self.BCE = nn.BCELoss().to(self.device)
        self.MSE = nn.MSELoss().to(self.device)
        self.CE = nn.CrossEntropyLoss().to(self.device)
        self.strides = torch.unsqueeze(self.strides, 1)
        self.anchors = self.anchors/self.strides             #[9, 2]
        # self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(self.device)
        # self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1])).to(self.device)

    def forward(self, prediction, gt):
        self.gt = gt
        batch_size, channel, width, height = prediction.size() #batch_size, (5+num_classes)*3, width, height
        #将anchor缩放到特征图里面
        # scale = self.inputwidth//width
        scale = self.strides[self.mask][0]
        anchorw = self.anchors[:, 0]            #[9]
        # anchor_w = torch.reshape(anchorw, (1, 9, 1, 1))  #(1, 9, 1, 1)
        anchorh = self.anchors[:, 1]            #[9]
        # anchor_h = torch.reshape(anchorh, (1, 9, 1, 1))  #(1, 9, 1, 1)

        mask_anchorw = anchorw[self.mask]
        mask_anchorh = anchorh[self.mask]
        mask_anchor_w = torch.reshape(mask_anchorw, (1, 3, 1, 1))  #(1, 3, 1, 1)
        mask_anchor_h = torch.reshape(mask_anchorh, (1, 3, 1, 1))  #(1, 3, 1, 1)
        
        ## num_classes = 20 batch_size = 2  prediction [2, 75, 13, 13]
        # prediction1 = torch.unsqueeze(prediction[:, :channel//3, :, :], 1)   #prediction.view((bs, 3, width, height, -1))#[2, 1, 25, 13, 13]
        # prediction2 = torch.unsqueeze(prediction[:, channel//3:channel*2//3, :, :], 1)     #[2, 1, 25, 13, 13]
        # prediction3 = torch.unsqueeze(prediction[:, channel*2//3:, :, :], 1)      #[2, 1, 25, 13, 13]
        # prediction = torch.cat([prediction1, prediction2, prediction3], dim=1)        #[2, 3, 25, 13, 13]
        ##x、y偏移量，w、h缩放值，confp置信度，classesp分类
        # cxp = self.sigmoid(prediction[:, :, 0, :, :])     #[2, 3, 13, 13]
        # cyp = self.sigmoid(prediction[:, :, 1, :, :])      #[2, 3, 13, 13]
        # wp = prediction[:, :, 2, :, :]     #[2, 3, 13, 13]
        # hp = prediction[:, :, 3, :, :]     #[2, 3, 13, 13]
        # confp = self.sigmoid(prediction[:, :, 4, :, :])     #[2, 3, 13, 13]
        # classesp = self.sigmoid(prediction[:, :, 5:, :, :])     #[2, 3, 20, 13, 13]
        # classesp = classesp.permute(0,1,3,4,2)     #[2, 3, 13, 13, 20]

        #prediction [2, 75, 13, 13]
        prediction = prediction.view((batch_size, 3, width, height, -1)) #[2, 3, 13, 13, 25]
        #x、y偏移量，w、h缩放值，confp置信度，classesp分类
        cxp = self.sigmoid(prediction[:, :, :, :, 0])     #[2, 3, 13, 13]
        cyp = self.sigmoid(prediction[:, :, :, :, 1])      #[2, 3, 13, 13]
        wp = prediction[:, :, :, :, 2]     #[2, 3, 13, 13]
        hp = prediction[:, :, :, :, 3]     #[2, 3, 13, 13]
        confp = self.sigmoid(prediction[:, :, :, :, 4])     #[2, 3, 13, 13]
        classesp = self.sigmoid(prediction[:, :, :, :, 5:])     #[2, 3, 13, 13, 20]

        # confp = torch.unsqueeze(confp, 4)    ##[2, 3, 13, 13, 1]
        # confp = confp*classesp             ##[2, 3, 13, 13, 20]

        #预测出来的坐标点的位置，加上偏移量即可
        x_coord = torch.arange(width).repeat(height, 1).to(self.device)    #[13, 13]
        y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).to(self.device)     #[13, 13]
        x_coord = torch.unsqueeze(torch.unsqueeze(x_coord, 0),0).to(self.device)    #[1, 1, 13, 13]
        y_coord = torch.unsqueeze(torch.unsqueeze(y_coord, 0),0).to(self.device)    #[1, 1, 13, 13]
        predx = cxp + x_coord      #[2, 3, 13, 13]
        predy = cyp + y_coord     #[2, 3, 13, 13]
        predw = torch.exp(wp)*mask_anchor_w     #[2, 3, 13, 13]
        predh = torch.exp(hp)*mask_anchor_h     #[2, 3, 13, 13]

        if len(self.gt)==0:
            return predx, predy, predw, predh, confp, classesp, 3, 3

        numpic = self.gt[:, -1]
        assert batch_size==numpic[-1] + 1

        num_anchors = mask_anchor_w.size()[1]

        gtctx = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        gtcty = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        gtw = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        gth = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)

        offsetctx = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        offsetcty = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        offsetw = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        offseth = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        objectness = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        smallscales = self.FloatTensor(batch_size, num_anchors, width, height).fill_(0)
        classes = self.FloatTensor(batch_size, num_anchors, width, height, self.num_classes).fill_(0)
        objmask = self.BoolTensor(batch_size, num_anchors, width, height).fill_(0)
        ignore_mask = self.BoolTensor(batch_size, num_anchors, width, height).fill_(0)

        #对于每个label计算和anchor之间IOU最匹配的特征层，不考虑中心点位置
        lossiou = 0
        glossiou = []

        #将真实label的框放到特征图里
        mapsize = 416/self.strides
        mapsize = mapsize[self.mask][0]
        iou_col = torch.zeros(0).to(self.device)
        ioulos_col = torch.zeros(0).to(self.device)
        num_correct = 0
        all_gtnums  = len(self.gt)
        for bs in range(batch_size):
            batch_gt = self.gt[self.gt[:, -1]==bs][:, :-1]
            for i, rectangle in enumerate(batch_gt):
                center_x = rectangle[1] * mapsize
                center_y = rectangle[2] * mapsize
                gtwidth  = rectangle[3] * mapsize
                gtheight = rectangle[4] * mapsize

                label = int(rectangle[0])
                gtxyxy = torch.zeros((1, 3 + 1)).to(self.device)    #xmin  ymin  xmax  ymax
                gtxyxy[:, 0] = center_x - gtwidth * 0.5
                gtxyxy[:, 1] = center_y - gtheight * 0.5
                gtxyxy[:, 2] = center_x + gtwidth * 0.5
                gtxyxy[:, 3] = center_y + gtheight * 0.5
                anchorxyxy = torch.zeros((3, 3+1)).to(self.device)
                anchorxyxy[:, 0] = center_x - self.anchors[self.mask, 0] * 0.5
                anchorxyxy[:, 1] = center_y - self.anchors[self.mask, 1] * 0.5
                anchorxyxy[:, 2] = center_x + self.anchors[self.mask, 0] * 0.5
                anchorxyxy[:, 3] = center_y + self.anchors[self.mask, 1] * 0.5
                iouanchor = box_iou(gtxyxy, anchorxyxy)[0]
                ioumask = iouanchor > 0.3

                gtctx[bs, ioumask, int(center_x), int(center_y)] = center_x
                gtcty[bs, ioumask, int(center_x), int(center_y)] = center_y
                gtw[bs, ioumask, int(center_x), int(center_y)] = gtwidth
                gth[bs, ioumask, int(center_x), int(center_y)] = gtheight
                objectness[bs, ioumask, int(center_x), int(center_y)] = 1
                # if(torch.sum(ioumask).item()==0):
                #     _, ind = torch.max(iouanchor, dim=0)
                #     objectness[bs, ind, int(center_x), int(center_y)] = 1
                objmask[bs, ioumask, int(center_x), int(center_y)] = True
                classes[bs, ioumask, int(center_x), int(center_y), label] = 1
                
                offsetctx[bs, ioumask, int(center_x), int(center_y)] = center_x - int(center_x)
                offsetcty[bs, ioumask, int(center_x), int(center_y)] = center_y - int(center_y)
                ofw = torch.log(gtwidth/mask_anchor_w[0, :, 0, 0][ioumask])
                ofh = torch.log(gtheight/mask_anchor_h[0, :, 0, 0][ioumask])
                offsetw[bs, ioumask, int(center_x), int(center_y)] = ofw.float()
                offseth[bs, ioumask, int(center_x), int(center_y)] = ofh.float()

                predict_x = predx[bs, ioumask, int(center_x), int(center_y)]
                predict_y = predy[bs, ioumask, int(center_x), int(center_y)]
                predict_w = predw[bs, ioumask, int(center_x), int(center_y)]
                predict_h = predh[bs, ioumask, int(center_x), int(center_y)]
                predict_boxes = torch.stack([predict_x, predict_y, predict_w, predict_h]).T
                predict_xyxy = torch.zeros(predict_boxes.size()).to(self.device)
                predict_xyxy[:, 0] = predict_boxes[:, 0] - predict_boxes[:, 2] * 0.5
                predict_xyxy[:, 1] = predict_boxes[:, 1] - predict_boxes[:, 3] * 0.5
                predict_xyxy[:, 2] = predict_boxes[:, 0] + predict_boxes[:, 2] * 0.5
                predict_xyxy[:, 3] = predict_boxes[:, 1] + predict_boxes[:, 3] * 0.5

                boxloss_scale =   2.0 - ((1.0 * (gtwidth * scale) * (gtheight * scale)) / self.areas)
                smallscales[bs, ioumask, int(center_x), int(center_y)] = boxloss_scale

                p_g_iou       =   box_iou(gtxyxy, predict_xyxy)[0]
                iou_col       =   torch.cat([iou_col, p_g_iou])

                iou_los       =   (1 - p_g_iou) * boxloss_scale
                ioulos_col    =   torch.cat([ioulos_col, iou_los])

                iou = torch.max(p_g_iou, dim = 0)[0].item()
                predlabel = torch.argmax(classesp[bs, ioumask, int(center_x), int(center_y)]).item()
                score = confp[bs, ioumask, int(center_x), int(center_y)][0].item()
                if ( iou > 0.5 and predlabel==label and score > 0.38):
                    num_correct += 1
        
        nProposals = int((confp > 0.5).sum().item())
        recall = float(num_correct/all_gtnums) if all_gtnums else 1
        precision = float(num_correct/(nProposals+1e-10))

        self.BCE_scale = nn.BCELoss(weight=smallscales).to(self.device)
        wp, offsetw = wp * smallscales, offsetw * smallscales
        hp, offseth = hp * smallscales, offseth * smallscales
        lossx         =  self.BCE_scale(cxp, offsetctx)
        lossy         =  self.BCE_scale(cyp, offsetcty)
        lossw         =  self.MSE(wp, offsetw)
        lossh         =  self.MSE(hp, offseth)
        lossobj       =  self.BCE(confp, objectness)
        lossclasses   =  self.BCE(classesp, classes)
        recall50 = torch.sum(iou_col>0.5)/(torch.sum(classes)+1e-10)
        recall75 = torch.sum(iou_col>0.75)/(torch.sum(classes)+1e-10)

        if(len(ioulos_col)==0):
            lossiou = torch.zeros(1).to(self.device)
        else:
            lossiou = torch.mean(ioulos_col)
        loss = (lossx + lossy + lossw + lossh) + lossobj + lossclasses + lossiou

        if(torch.sum(objmask).item()==0):
            objectnessscore = 0
        else:
            objectnessscore = torch.mean(confp[objmask]).item()
        noobjectness = torch.mean(confp[~objmask]).item()

        return loss, objectnessscore, recall50.item(), recall75.item(), noobjectness, lossiou, recall, precision

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
        result3, objectness3, recall50_3, recall75_3, noobjectness3, lossiou3, recall3, precision3 = self.yolo3(y3, deepcopy(self.gt))

        #进入header层并上采样
        r2up = torch.cat([up3, upy2], dim=1)
        up2, y2 = self.hb2(r2up) #y2 [2, 255, 26, 26]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp
        result2, objectness2, recall50_2, recall75_2, noobjectness2, lossiou2, recall2, precision2 = self.yolo2(y2, deepcopy(self.gt))

        #进入header层
        r1up = torch.cat([up2, upy1], dim=1)
        y1 = self.hb1(r1up) #y1 [2, 255, 52, 52]
        #当训练时即len(self.gt)==0 返回的就是loss、准确率、召回率、~、预测置信度、预测类别
        #测试或者detect时，就是predx, predy, predw, predh, confp, classesp
        result1, objectness1, recall50_1, recall75_1, noobjectness1, lossiou1, recall1, precision1 = self.yolo1(y1, deepcopy(self.gt))

        #[2, 255, 13, 13]、[2, 255, 26, 26]、[2, 255, 52, 52]
        if len(self.gt)!=0:
            #训练时返回值
            length = len(np.where(np.array([objectness3,objectness2,objectness1])!=0)[0])
            # print(objectness3, objectness2, objectness1, length)
            objectness = (objectness3 + objectness2 + objectness1)/(length + 1e-10)
            noobjectness = (noobjectness3 + noobjectness2 + noobjectness1)/(length + 1e-10)
            recall50 = (recall50_1 + recall50_2 + recall50_3)/(length + 1e-10)
            recall75 = (recall75_1 + recall75_2 + recall75_3)/(length + 1e-10)
            recall = (recall3 + recall2 + recall1)/3
            precision = (precision3 + precision2 + precision1)/3
            return result3, result2, result1, objectness, recall50, recall75, noobjectness, recall, precision
        else:
            #检测或者测试时返回值
            results = [[result3, objectness3, recall50_3, recall75_3, noobjectness3, lossiou3]\
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
                mask = scoresmaxp>self.score_thresh    ##[2, 3, 13, 13]

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
            prediction = np.array(prediction)
            return prediction, 2,  2,  2,  2,  2, 2, 2, 2, 2

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
