#Author：ZouJiu
#Time: 20230727

import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import cv2
import time
import torch
from copy import deepcopy 
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from models.layer_yolo import yololayer

class inputnet(nn.Module):
    def __init__(self):
        super(inputnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=True)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu0 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
       x = self.leaky_relu0(self.norm1(self.conv1(x)))
       x = self.leaky_relu1(self.norm2(self.conv2(x)))
       return x

class resblock1(nn.Module):
    def __init__(self):
        super(resblock1, self).__init__()
        self.conv1 = nn.Conv2d(64, 32, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(64)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        # self.relu = nn.ReLU()

        self.conv_extra = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True)
        self.norm_extra = nn.BatchNorm2d(128)

    def forward(self, x):
        y = self.leaky_relu1(self.norm1(self.conv1(x)))
        y = self.leaky_relu2(self.norm2(self.conv2(y)))
        y = x + y
        y = self.leaky_relu3(self.norm_extra(self.conv_extra(y)))

        return y

class resblock2_child(nn.Module):
    def __init__(self):
        super(resblock2_child, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(128)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu1(self.norm1(self.conv1(x)))
        y = self.leaky_relu2(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock2(nn.Module):
    def __init__(self):
        super(resblock2, self).__init__()
        self.resblock2_child = nn.ModuleList([resblock2_child() for i in range(2)])
        self.conv_extra = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True)
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
        self.conv1 = nn.Conv2d(256, 128, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(256)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu1(self.norm1(self.conv1(x)))
        y = self.leaky_relu2(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock3(nn.Module):
    def __init__(self):
        super(resblock3, self).__init__()
        self.resblock3_child = nn.ModuleList([resblock3_child() for i in range(8)])
        self.conv_extra = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True)
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
        self.conv1 = nn.Conv2d(512, 256, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(512)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        y = self.leaky_relu1(self.norm1(self.conv1(x)))
        y = self.leaky_relu2(self.norm2(self.conv2(y)))
        y = x + y
        return y

class resblock4(nn.Module):
    def __init__(self):
        super(resblock4, self).__init__()
        self.resblock4_child = nn.ModuleList([resblock4_child() for i in range(8)])
        self.conv_extra = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=True)
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
        self.conv1 = nn.Conv2d(1024, 512, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(1024)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        y = self.leaky_relu1(self.norm1(self.conv1(x)))
        y = self.leaky_relu2(self.norm2(self.conv2(y)))
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
        self.conv1 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(beginchannel)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu66 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm4 = nn.BatchNorm2d(beginchannel)
        self.conv5 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm5 = nn.BatchNorm2d(channel)

        #yolo1,接conv5
        self.conv7 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm7 = nn.BatchNorm2d(beginchannel)
        self.conv8 = nn.Conv2d(beginchannel, (5+num_classes)*3, 1, stride=1, bias=True)
        # self.relu = nn.ReLU()

        #upsample,接conv5
        self.conv9 = nn.Conv2d(channel, channel//2, 1, stride=1, bias=True)
        self.norm9 = nn.BatchNorm2d(channel//2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.leaky_relu1(self.norm1(self.conv1(x)))
        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        x = self.leaky_relu5(self.norm5(self.conv5(x)))

        y3 = self.leaky_relu6(self.norm7(self.conv7(x)))
        y3 = self.conv8(y3)
        up3 = self.leaky_relu66(self.norm9(self.conv9(x)))
        up3 = self.upsample(up3)
        return up3, y3


class header_block_finally(nn.Module):
    def __init__(self, num_classes, beginchannel, channel, upornot=True):
        super(header_block_finally, self).__init__()
        self.upornot = upornot
        self.conv1 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm1 = nn.BatchNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm2 = nn.BatchNorm2d(beginchannel)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu4 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu5 = nn.LeakyReLU(negative_slope=0.1)
        self.leaky_relu6 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm3 = nn.BatchNorm2d(channel)
        self.conv4 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm4 = nn.BatchNorm2d(beginchannel)
        self.conv5 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=True)
        self.norm5 = nn.BatchNorm2d(channel)

        #yolo1,接conv5
        self.conv7 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=True)
        self.norm7 = nn.BatchNorm2d(beginchannel)
        self.conv8 = nn.Conv2d(beginchannel, (5+num_classes)*3, 1, stride=1, bias=True)

    def forward(self, x):
        x = self.leaky_relu1(self.norm1(self.conv1(x)))
        x = self.leaky_relu2(self.norm2(self.conv2(x)))
        x = self.leaky_relu3(self.norm3(self.conv3(x)))
        x = self.leaky_relu4(self.norm4(self.conv4(x)))
        x = self.leaky_relu5(self.norm5(self.conv5(x)))

        y3 = self.leaky_relu6(self.norm7(self.conv7(x)))
        y3 = self.conv8(y3)
        return y3

class yolov3_backbone(nn.Module):
    def __init__(self, num_classes):
        super(yolov3_backbone, self).__init__()
        self.num_classes = num_classes
        self.net = inputnet()
        self.ResBlock1 = resblock1()
        self.ResBlock2 = resblock2()
        self.ResBlock3 = resblock3()
        self.ResBlock4 = resblock4()
        self.ResBlock5 = resblock5()

        self.hb3 = header_block(self.num_classes, 1024, 512, upornot=True) #r3.size()[1] 第二个参数  最深层
        self.hb2 = header_block(self.num_classes, 768, 256, upornot=True) #r2up.size()[1] 第二个参数  中等层
        self.hb1 = header_block_finally(self.num_classes, 384, 128, upornot=False) #r1up.size()[1] 第二个参数  最浅层

    def forward(self, x):
        r = self.net(x)
        r = self.ResBlock1(r)
        r = self.ResBlock2(r)
        upy1, r1  = self.ResBlock3(r)
        upy2, r2 = self.ResBlock4(r1)
        r3 = self.ResBlock5(r2)
        
        #进入header层并上采样    大目标  特征图缩放尺度大网络深，特征图本身尺寸小 16x16
        up3, y3 = self.hb3(r3) #y3 [2, 255, 16, 16]

        #进入header层并上采样    中等的目标  特征图缩放尺度中等网络中部，特征图本身尺寸中等 32x32
        r2up = torch.cat([up3, upy2], dim=1)
        up2, y2 = self.hb2(r2up) #y2 [2, 255, 32, 32]

        #进入header层    小目标  特征图缩放尺度小网络浅层，特征图本身尺寸大 64x64
        r1up = torch.cat([up2, upy1], dim=1)
        y1 = self.hb1(r1up) #y1 [2, 255, 64, 64]
        
        return y1, y2, y3                 ##  small obj       middle obj         big obj

class Yolov3Net(nn.Module):
    def __init__(self, num_classes, anchors, device, imgsize):
        super(Yolov3Net, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.imgsize = imgsize
        self.anchors = anchors
        self.yolo = [yololayer(self.device, len(self.anchors[0]), self.num_classes) for _ in range(len(anchors))]
        self.anchors_sparse = [torch.tensor(anchors[i]).float().view(-1, 2).view(1, -1, 1, 1, 2).to(device) for i in range(len(anchors))]
        self.yolov3Net_backbone = yolov3_backbone(num_classes)

    def forward(self, x, yolovfive = False):
        out = self.yolov3Net_backbone(x)                   #            [small obj   middle obj   big obj]
        prediction = [self.yolo[i](out[i], self.anchors_sparse[i], self.imgsize, yolovfive = yolovfive) for i in range(len(out))]
        if self.training:
            return prediction # prediction, anchors
        else:
            kee = []
            for i in range(len(prediction)):
                kee.append(prediction[i][0])
            return torch.cat(kee, 1)

# def iou_p_g(groundtruth, predict):
#     cx, cy, w, h = groundtruth[:, 0], groundtruth[:, 1],groundtruth[:, 2],groundtruth[:, 3]
#     cxp, cyp, wp, hp = predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3]
#     xmin = cx - w/2
#     ymin = cy - h/2
#     xmax = cx + w/2
#     ymax = cy + h/2

#     xminp = cxp - wp/2
#     yminp = cyp - hp/2
#     xmaxp = cxp + wp/2
#     ymaxp = cyp + hp/2
#     alliou = torch.zeros((predict.size()[0], groundtruth.size()[0]))
#     for i in range(groundtruth.size()[0]):
#         singleiou = []
#         join = (torch.min(xmax[i], xmaxp) - torch.max(xmin[i], xminp)).clamp(0)*\
#             (torch.min(ymax[i], ymaxp) - torch.max(ymin[i], yminp)).clamp(0)
#         gtarea = (xmax[i] - xmin[i])*(ymax[i] - ymin[i])
#         parea = (xmaxp - xminp)*(ymaxp - yminp)
#         iouresult = join/(gtarea+parea-join)
#         alliou[:, i] = iouresult
#     return alliou

# def iou_box(groundtruth, predict):
#     cx, cy, w, h = groundtruth[:, 0], groundtruth[:, 1],groundtruth[:, 2],groundtruth[:, 3]
#     cxp, cyp, wp, hp = predict[:, 0], predict[:, 1], predict[:, 2], predict[:, 3]
#     xmin = cx - w/2
#     ymin = cy - h/2
#     xmax = cx + w/2
#     ymax = cy + h/2

#     xminp = cxp - wp/2
#     yminp = cyp - hp/2
#     xmaxp = cxp + wp/2
#     ymaxp = cyp + hp/2

#     join = (torch.min(xmax, xmaxp) - torch.max(xmin, xminp))*\
#         (torch.min(ymax, ymaxp) - torch.max(ymin, yminp))
#     gtarea = (xmax - xmin)*(ymax - ymin)
#     parea = (xmaxp - xminp)*(ymaxp - yminp)
#     iouresult = join/(gtarea+parea-join)
#     return iouresult

# def iouwh(batch_gt00wh, anchor00wh): 
#     #定义两者的中心点坐标重合，只要考虑长和宽，不妨定中心点(x, y) = (90, 90)，实际不会是0
#     #做了inner全连接
#     assume = torch.ones(batch_gt00wh.size()[0])*90
#     if torch.cuda.is_available():
#         assume = assume.to("cuda")
#     gtw, gth = batch_gt00wh[:, 0], batch_gt00wh[:, 1]
#     anchorw, anchorh = anchor00wh[:, 0], anchor00wh[:, 1]
#     xmingt = assume - gtw/2
#     ymingt = assume - gth/2
#     xmaxgt = assume + gtw/2
#     ymaxgt = assume + gth/2

#     xminan = assume - anchorw/2
#     yminan = assume - anchorh/2
#     xmaxan = assume + anchorw/2
#     ymaxan = assume + anchorh/2
#     join = (torch.min(xmaxgt, xmaxan) - torch.max(xmingt, xminan)).clamp(0)*\
#         (torch.min(ymaxgt, ymaxan) - torch.max(ymingt, yminan)).clamp(0)
#     gtarea = (xmaxgt - xmingt)*(ymaxgt - ymingt)
#     anchorarea = (xmaxan - xminan)*(ymaxan - yminan)
#     iouresult = join/(gtarea+anchorarea-join)
#     length = iouresult.size()[0]//9
#     #对全连接进行合并
#     res1 = torch.unsqueeze(iouresult[:length], 1)
#     res2 = torch.unsqueeze(iouresult[length:length*2], 1)
#     res3 = torch.unsqueeze(iouresult[length*2:length*3], 1)
#     res4 = torch.unsqueeze(iouresult[length*3:length*4], 1)
#     res5 = torch.unsqueeze(iouresult[length*4:length*5], 1)
#     res6 = torch.unsqueeze(iouresult[length*5:length*6], 1)
#     res7 = torch.unsqueeze(iouresult[length*6:length*7], 1)
#     res8 = torch.unsqueeze(iouresult[length*7:length*8], 1)
#     res9 = torch.unsqueeze(iouresult[length*8:], 1)
#     result = torch.cat([res1, res2, res3,\
#         res4, res5, res6,\
#         res7, res8, res9], dim=1) #[gt的数量, 3]，3代表3个anchor，也就是特征图的3层
#     return result

# def nms(predict, nms_thresh):
#     #[cx, cy, w, h, maxscore, label]
#     if len(predict)==0:
#         return []
#     index  = np.argsort(predict[:, 4])
#     index = list(index)
#     index.reverse()
#     predict = predict[index]
#     xmin = predict[:, 0] - predict[:, 2]/2
#     ymin = predict[:, 1] - predict[:, 3]/2
#     xmax = predict[:, 0] + predict[:, 2]/2
#     ymax = predict[:, 1] + predict[:, 3]/2
#     areas = (ymax - ymin)*(xmax - xmin)
#     labeles = np.unique(predict[:, 5])
#     keep = []
#     # print(predict, predict.shape)
#     for j in range(len(labeles)):
#         ind = np.where(predict[:, 5]==labeles[j])[0]
#         if len(ind)==0:
#             continue
#         # if len(ind)!=1:
#         #     print(ind)
#         while len(ind)>0:
#             i = ind[0]
#             keep.append(i)

#             x1min = np.maximum(xmin[i], xmin[ind[1:]])
#             y1min = np.maximum(ymin[i], ymin[ind[1:]])
#             x1max = np.minimum(xmax[i], xmax[ind[1:]])
#             y1max = np.minimum(ymax[i], ymax[ind[1:]])
#             overlap = np.maximum(0, (y1max-y1min))*np.maximum(0, (x1max-x1min))

#             ioures = overlap/(areas[i] + areas[ind[1:]] - overlap)
#             # t = np.where(ioures <= nms_thresh)[0]
#             maskiou = ioures<= nms_thresh
#             if len(maskiou)==0:
#                 break
#             # print(1111111, ind)
#             ind = ind[1:][ioures <= nms_thresh]
#             # print(ioures <= nms_thresh, ind)
#     # print(3333333, keep)
#     return predict[keep]

# def box_iou(box1, box2):
#     # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
#     """
#     Return intersection-over-union (Jaccard index) of boxes.
#     Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
#     Arguments:
#         box1 (Tensor[N, 4])
#         box2 (Tensor[M, 4])
#     Returns:
#         iou (Tensor[N, M]): the NxM matrix containing the pairwise
#             IoU values for every element in boxes1 and boxes2
#     """

#     def box_area(box):
#         # box = 4xn
#         return (box[2] - box[0]) * (box[3] - box[1])

#     area1 = box_area(box1.T)
#     area2 = box_area(box2.T)

#     # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
#     inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
#     return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

# def GIOU_xywh_torch(boxes1, boxes2):
#     """
#      https://arxiv.org/abs/1902.09630
#      https://github.com/Peterisfar/YOLOV3/blob/master/utils/tools.py#L199
#     boxes1(boxes2)' shape is [..., (x,y,w,h)].The size is for original image.
#     """
#     # xywh->xyxy
#     boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
#                         boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
#     boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
#                         boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

#     boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
#                         torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
#     boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
#                         torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

#     boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
#     boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

#     inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
#     inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
#     inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
#     inter_area =  inter_section[..., 0] * inter_section[..., 1]
#     union_area = boxes1_area + boxes2_area - inter_area
#     IOU = 1.0 * inter_area / union_area

#     enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
#     enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
#     enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
#     enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

#     GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
#     return GIOU


# def iou_xywh_torch(boxes1, boxes2):
#     """
#     :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
#     :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
#     :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
#     """
#     boxes1_area = boxes1[..., 2] * boxes1[..., 3]
#     boxes2_area = boxes2[..., 2] * boxes2[..., 3]

#     # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
#     # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
#     boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
#                         boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
#     boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
#                         boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

#     # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
#     left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
#     right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

#     # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
#     inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#     union_area = boxes1_area + boxes2_area - inter_area
#     IOU = 1.0 * inter_area / union_area
#     return IOU

# class yolo(nn.Module):
#     '''
#     对输出的特征图进行解码，得到坐标、置信度、分类的概率
#     并计算相应的objectness loss、classify loss、coordinates loss
#     '''
#     def __init__(self, anchors, stride, mask, device):
#         super(yolo, self).__init__()
#         self.FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
#         self.LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
#         self.anchors = self.FloatTensor(anchors)
#         self.ori_anchors = self.FloatTensor(anchors)
#         self.stride = stride
#         self.mask = mask
#         self.device = device

#         self.sigmoid = nn.Sigmoid()
#         self.anchors_g = self.anchors[self.mask]/self.stride

#     def forward(self, prediction):
#         batch_size, channel, width, height = prediction.size() #batch_size, (5+num_classes)*3, width, height

#         #prediction [2, 75, 13, 13]
#         prediction = prediction.view((batch_size, width, height, 3, -1)) #[2, 13, 13, 3, 25]
#         #x、y偏移量，w、h缩放值，confp置信度，classesp分类
#         cxp = self.sigmoid(prediction[:, :, :, :, 0]).to(self.device)     #[2, 13, 13, 3]
#         cyp = self.sigmoid(prediction[:, :, :, :, 1]).to(self.device)      #[2, 13, 13, 3]
#         wp = prediction[:, :, :, :, 2].to(self.device)     #[2, 3, 13, 13]
#         hp = prediction[:, :, :, :, 3].to(self.device)     #[2, 3, 13, 13]
#         confp = self.sigmoid(prediction[:, :, :, :, 4]).to(self.device)     #[2, 13, 13, 3]
#         classesp = self.sigmoid(prediction[:, :, :, :, 5:]).to(self.device)     #[2, 13, 13, 3, 20]

#         #预测出来的坐标点的位置，加上偏移量即可
#         mask_anchor_w = torch.reshape(self.anchors_g[:, 0], (1, 1, 1, 3))  #(1, 1, 1, 3)
#         mask_anchor_h = torch.reshape(self.anchors_g[:, 1], (1, 1, 1, 3))  #(1, 1, 1, 3)
#         x_coord = torch.arange(width).repeat(height, 1).to(self.device)    #[13, 13]
#         y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).to(self.device)     #[13, 13]
#         x_coord = torch.unsqueeze(torch.unsqueeze(x_coord, 0),3).to(self.device)    #[1, 13, 13, 1]
#         y_coord = torch.unsqueeze(torch.unsqueeze(y_coord, 0),3).to(self.device)    #[1, 13, 13, 1]
#         predx = ((cxp + x_coord) * self.stride).to(self.device)    #放到原图尺寸416x416  #[2, 13, 13, 3]
#         predy = ((cyp + y_coord) * self.stride).to(self.device)     #[2, 13, 13, 3]
#         predw = ((torch.exp(wp) * mask_anchor_w) * self.stride).to(self.device)      #[2, 13, 13, 3]
#         predh = ((torch.exp(hp) * mask_anchor_h) * self.stride).to(self.device)     #[2, 13, 13, 3]

#         return predx, predy, predw, predh, confp, classesp, \
#                cxp, cyp, wp, hp, prediction[:, :, :, :, 4], prediction[:, :, :, :, 5:]

# class lossyolo(nn.Module):
#     def __init__(self, iouthresh, ignore_threshold, stride, device, inputwidth):
#         super(lossyolo, self).__init__()
#         self.stride = stride
#         self.ignore_threshold = ignore_threshold
#         self.iouthresh = iouthresh
#         self.device = device
#         self.inputwidth = inputwidth
#         self.BCElog = nn.BCEWithLogitsLoss(reduction='none').to(self.device)
#         self.MSE = nn.MSELoss(reduction='none').to(self.device)

#     def forward(self, label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox, \
#                            p, p_d):
#         L_loss, L_loss_giou, L_loss_conf, L_loss_cls, Lrecall50, Lrecall75, Lnum_ins, Lobj, Lnoobj = \
#             self.__cal_per_layer_loss(label_lbbox, lbbox, p[0], p_d[0], self.stride[0])
#         M_loss, M_loss_giou, M_loss_conf, M_loss_cls, Mrecall50, Mrecall75, Mnum_ins, Mobj, Mnoobj = \
#             self.__cal_per_layer_loss(label_mbbox, mbbox, p[1], p_d[1], self.stride[1])
#         S_loss, S_loss_giou, S_loss_conf, S_loss_cls, Srecall50, Srecall75, Snum_ins, Sobj, Snoobj = \
#             self.__cal_per_layer_loss(label_sbbox, sbbox, p[2], p_d[2], self.stride[2])

#         loss = L_loss + M_loss + S_loss
#         loss_giou = L_loss_giou + M_loss_giou + S_loss_giou
#         loss_conf = L_loss_conf + M_loss_conf + S_loss_conf
#         loss_cls  = L_loss_cls  + M_loss_cls  + S_loss_cls

#         ins = (Lnum_ins + Mnum_ins + Snum_ins)
#         recall50  = (Lrecall50 + Mrecall50 + Srecall50)/ins
#         recall75  = (Lrecall75 + Mrecall75 + Srecall75)/ins
#         obj = (Lobj + Mobj + Sobj)/ins
#         noobj = (Lnoobj + Mnoobj + Snoobj)/3

#         return loss, loss_giou.item(), loss_conf.item(), loss_cls.item(), \
#             recall50.item(), recall75.item(), obj.item(), noobj
        
#     def __cal_per_layer_loss(self, label_bbox, bbox, p, p_d, stride):
#         label_bbox = label_bbox.to(self.device)
#         bbox = bbox.to(self.device)
#         # predx, predy, predw, predh, confp, classesp, \
#         #             cxp, cyp, wp, hp, confp_ori, classesp_ori = predict
#         batch_size, grid = p.shape[:2]
#         imgsize = grid * stride

#         assert imgsize==self.inputwidth

#         confp_ori = p[..., 4:5]
#         classesp_ori = p[..., 5:]

#         predxywh = p_d[..., :4]
#         confp = p_d[..., 4]

#         label_xywh = label_bbox[..., :4]
#         label_obj_mask = label_bbox[..., 4]
#         label_cls  = label_bbox[..., 5:]

#         giou = GIOU_xywh_torch(predxywh, label_xywh)
#         bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2] * label_xywh[..., 3] / (imgsize ** 2)
#         lossgiou = label_obj_mask * bbox_loss_scale * (1.0 - giou)
#         lossgiou = lossgiou.unsqueeze(-1)

#         iou = iou_xywh_torch(predxywh.unsqueeze(4), bbox.unsqueeze(1).unsqueeze(1).unsqueeze(1))
#         ioumax = iou.max(-1, keepdim=True)[0].squeeze(-1)
#         label_noobj_mask = (1 - label_obj_mask) * (ioumax < self.ignore_threshold).float()
#         label_noobj_mask.to(self.device)

#         midconf = self.BCElog(confp_ori.squeeze(), label_obj_mask) * (torch.pow(torch.abs(label_obj_mask - torch.sigmoid(confp_ori.squeeze())), 2))
#         lossconf = label_obj_mask * midconf + label_noobj_mask * midconf

#         losscls = label_obj_mask.unsqueeze(-1) * self.BCElog(classesp_ori.squeeze(), label_cls)

#         loss_giou = torch.sum(lossgiou)/batch_size
#         loss_conf = torch.sum(lossconf)/batch_size
#         loss_cls  = torch.sum(losscls)/batch_size

#         loss = loss_giou + loss_conf + loss_cls

#         Recall = label_obj_mask * ioumax
#         recall50 = torch.sum(Recall > 0.5)
#         recall75 = torch.sum(Recall > 0.75)
#         num_ins = torch.sum(label_obj_mask)
        
#         noobj = torch.mean(label_noobj_mask * confp).item()
#         if(torch.sum(label_obj_mask).item()==0):
#             obj = 0
#         else:
#             obj = torch.sum(label_obj_mask * confp).item()
#         return loss, loss_giou, loss_conf, loss_cls, recall50, recall75, num_ins, obj, noobj

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

    yolov3 = Yolov3(num_classes, anchors, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
    result3, result2, result1, precision50, recall50, recall75 = yolov3(image, gt)
    print(result3, result2, result1, precision50, recall50, recall75)
