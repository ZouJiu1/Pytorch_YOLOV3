#Author：ZouJiu
#Time: 2022-12-10

import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import torch
import torch.nn as nn
from models.layer_yolo import yololayer

class ConvBlock_LN(nn.Module):
    def __init__(self, inc, ouc, kernel_size, stride, padding, groups = 1, bias = False, rel=True):
        super(ConvBlock_LN, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv = nn.Conv2d(inc, ouc, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn   = nn.BatchNorm2d(ouc)
        self.rel = rel
    def forward(self, x):
        if self.rel:
            x = self.relu(self.bn(self.conv(x)))
        else:
            x = self.bn(self.conv(x))
        return x

class yolov3tiny_backbone(nn.Module):
    def __init__(self, classes):
        super(yolov3tiny_backbone, self).__init__()
        self.maxpool0 = nn.MaxPool2d(2, 2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.maxpool6 = nn.MaxPool2d(2, 2)
        self.maxpool7 = nn.MaxPool2d(2, 1)
        self.block1 = ConvBlock_LN(3, 16, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block2 = ConvBlock_LN(16, 32, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block3 = ConvBlock_LN(32, 64, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block4 = ConvBlock_LN(64, 128, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block5 = ConvBlock_LN(128, 256, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block6 = ConvBlock_LN(256, 512, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block7 = ConvBlock_LN(512, 1024, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block8 = ConvBlock_LN(1024, 256, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)
        self.block9 = ConvBlock_LN(256, 512, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block10 = nn.Conv2d(512, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = True)

        self.block11 = ConvBlock_LN(256, 128, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = False)

        self.block12 = ConvBlock_LN(384, 256, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block13 = nn.Conv2d(256, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = True)
        self.upsample = nn.Upsample(scale_factor=2)
        self.padding = nn.ZeroPad2d((0, 1, 0, 1))

    def forward(self, x):
        x = self.maxpool1(self.block2(self.maxpool0(self.block1(x))))
        x = self.maxpool3(self.block4(self.maxpool2(self.block3(x))))
        y1 = self.block5(x)
        
        x = self.block6(self.maxpool6(y1))
        x = self.maxpool7(self.padding(x))
        x = self.block7(x)
        y2 = self.block8(x)
        out1 =  self.block10(self.block9(y2))
        
        y3 = self.block11(y2)
        x = self.upsample(y3)
        x = torch.concat([x, y1], dim = 1)
        out2 = self.block13(self.block12(x))

        return out2, out1 #out2 small obj   out1 big obj


class yolov3tiny_backbone_depthwise(nn.Module):
    def __init__(self, classes):
        super(yolov3tiny_backbone_depthwise, self).__init__()
        self.block1 = ConvBlock_LN(3, 16, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.maxpool0 = nn.MaxPool2d(2, 2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.maxpool3 = nn.MaxPool2d(2, 2)
        self.maxpool6 = nn.MaxPool2d(2, 2)
        self.maxpool7 = nn.MaxPool2d(2, 1)
        self.block2 = ConvBlock_LN(16, 32, kernel_size = 3, stride = 1, padding = 1, groups = 1, bias = False)
        self.block3 = ConvBlock_LN(32, 64, kernel_size = 3, stride = 1, padding = 1, groups = 32, bias = False)
        self.block4 = ConvBlock_LN(64, 128, kernel_size = 3, stride = 1, padding = 1, groups = 64, bias = False)
        self.block5 = ConvBlock_LN(128, 256, kernel_size = 3, stride = 1, padding = 1, groups = 128, bias = False)
        self.block6 = ConvBlock_LN(256, 512, kernel_size = 3, stride = 1, padding = 1, groups = 256, bias = False)
        self.block7 = ConvBlock_LN(512, 1024, kernel_size = 3, stride = 1, padding = 1, groups = 512, bias = False)
        self.block8 = ConvBlock_LN(1024, 256, kernel_size = 1, stride = 1, padding = 0, groups = 256, bias = False)
        self.block9 = ConvBlock_LN(256, 512, kernel_size = 3, stride = 1, padding = 1, groups = 256, bias = False)
        self.block10 = nn.Conv2d(512, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = True)

        self.block11 = ConvBlock_LN(256, 128, kernel_size = 1, stride = 1, padding = 0, groups = 128, bias = False)

        self.block12 = ConvBlock_LN(384, 256, kernel_size = 3, stride = 1, padding = 1, groups = 32, bias = False)
        self.block13 = nn.Conv2d(256, (classes+5) * 3, kernel_size = 1, stride = 1, padding = 0, groups = 1, bias = True)
        self.upsample = nn.Upsample(scale_factor=2)
        # self.padding = nn.ZeroPad2d((0, 1, 0, 1))

    def forward(self, x):
        x = self.maxpool1(self.block2(self.maxpool0(self.block1(x))))
        x = self.maxpool3(self.block4(self.maxpool2(self.block3(x))))
        y1 = self.block5(x)
        
        x = self.block6(self.maxpool6(y1))
        x = self.maxpool7(self.padding(x))
        x = self.block7(x)
        y2 = self.block8(x)
        out1 =  self.block10(self.block9(y2))
        
        y3 = self.block11(y2)
        x = self.upsample(y3)
        x = torch.concat([x, y1], dim = 1)
        out2 = self.block13(self.block12(x))

        return out2, out1 #out2 small obj   out1 big obj

class yolov3tinyNet(nn.Module):
    def __init__(self, num_classes, anchors, device, imgsize):
        super(yolov3tinyNet, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.imgsize = imgsize
        self.anchors = anchors
        self.yolo0 = yololayer(self.device, len(self.anchors[0]), self.num_classes)
        self.yolo1 = yololayer(self.device, len(self.anchors[0]), self.num_classes)
        self.yolo = [self.yolo0, self.yolo1]
        self.anchors_sparse = [torch.tensor(anchors[i]).float().view(-1, 2).view(1, -1, 1, 1, 2).to(device) for i in range(len(anchors))]
        self.yolov3tinyNet_backbone = yolov3tiny_backbone(num_classes)
        # self.yolov3tinyNet_backbone = yolov3tiny_backbone_depthwise(num_classes)

    def forward(self, x):
        out = self.yolov3tinyNet_backbone(x)   #[small obj, big obj, ...]
        prediction = [self.yolo[i](out[i], self.anchors_sparse[i], self.imgsize) for i in range(len(out))]
        if self.training:
            return prediction # prediction, anchors
        else:
            kee = []
            for i in range(len(prediction)):
                kee.append(prediction[i][0])
            return torch.cat(kee, 1)

# class inputnet(nn.Module):
#     def __init__(self):
#         super(inputnet, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(64)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

#     def forward(self, x):
#        x = self.leaky_relu(self.norm1(self.conv1(x)))
#        x = self.leaky_relu(self.norm2(self.conv2(x)))
#        return x

# class resblock1(nn.Module):
#     def __init__(self):
#         super(resblock1, self).__init__()
#         self.conv1 = nn.Conv2d(64, 32, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(32)

#         self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(64)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         self.relu = nn.ReLU()
        
#         self.conv_extra = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
#         self.norm_extra = nn.BatchNorm2d(128)


#     def forward(self, x):
#         y = self.leaky_relu(self.norm1(self.conv1(x)))
#         y = self.leaky_relu(self.norm2(self.conv2(y)))
#         y = x + y
#         y = self.leaky_relu(self.norm_extra(self.conv_extra(y)))

#         return y

# class resblock2_child(nn.Module):
#     def __init__(self):
#         super(resblock2_child, self).__init__()
#         self.conv1 = nn.Conv2d(128, 64, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(64)

#         self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(128)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         #self.relu = nn.ReLU()

#     def forward(self, x):
#         y = self.leaky_relu(self.norm1(self.conv1(x)))
#         y = self.leaky_relu(self.norm2(self.conv2(y)))
#         y = x + y
#         return y

# class resblock2(nn.Module):
#     def __init__(self):
#         super(resblock2, self).__init__()
#         self.resblock2_child = nn.ModuleList([resblock2_child() for i in range(2)])
#         self.conv_extra = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
#         self.norm_extra = nn.BatchNorm2d(256)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

#     def forward(self, x):
#         for i, rb2 in enumerate(self.resblock2_child):
#             x = rb2(x)
#         x = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
#         return x

# class resblock3_child(nn.Module):
#     def __init__(self):
#         super(resblock3_child, self).__init__()
#         self.conv1 = nn.Conv2d(256, 128, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(128)

#         self.conv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(256)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         #self.relu = nn.ReLU()

#     def forward(self, x):
#         y = self.leaky_relu(self.norm1(self.conv1(x)))
#         y = self.leaky_relu(self.norm2(self.conv2(y)))
#         y = x + y
#         return y

# class resblock3(nn.Module):
#     def __init__(self):
#         super(resblock3, self).__init__()
#         self.resblock3_child = nn.ModuleList([resblock3_child() for i in range(8)])
#         self.conv_extra = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
#         self.norm_extra = nn.BatchNorm2d(512)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

#     def forward(self, x):
#         for i, rb2 in enumerate(self.resblock3_child):
#             x = rb2(x)
#         y = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
#         return x,y

# class resblock4_child(nn.Module):
#     def __init__(self):
#         super(resblock4_child, self).__init__()
#         self.conv1 = nn.Conv2d(512, 256, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(256)

#         self.conv2 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(512)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         #self.relu = nn.ReLU()

#     def forward(self, x):
#         y = self.leaky_relu(self.norm1(self.conv1(x)))
#         y = self.leaky_relu(self.norm2(self.conv2(y)))
#         y = x + y
#         return y

# class resblock4(nn.Module):
#     def __init__(self):
#         super(resblock4, self).__init__()
#         self.resblock4_child = nn.ModuleList([resblock4_child() for i in range(8)])
#         self.conv_extra = nn.Conv2d(512, 1024, 3, stride=2, padding=1, bias=False)
#         self.norm_extra = nn.BatchNorm2d(1024)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

#     def forward(self, x):
#         for i, rb2 in enumerate(self.resblock4_child):
#             x = rb2(x)
#         y = self.leaky_relu(self.norm_extra(self.conv_extra(x)))
#         return x, y

# class resblock5_child(nn.Module):
#     def __init__(self):
#         super(resblock5_child, self).__init__()
#         self.conv1 = nn.Conv2d(1024, 512, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(512)

#         self.conv2 = nn.Conv2d(512, 1024, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(1024)

#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         #self.relu = nn.ReLU()

#     def forward(self, x):
#         y = self.leaky_relu(self.norm1(self.conv1(x)))
#         y = self.leaky_relu(self.norm2(self.conv2(y)))
#         y = x + y
#         return y

# class resblock5(nn.Module):
#     def __init__(self):
#         super(resblock5, self).__init__()
#         self.resblock5_child = nn.ModuleList([resblock5_child() for i in range(4)])

#     def forward(self, x):
#         for i, rb2 in enumerate(self.resblock5_child):
#             x = rb2(x)
#         return x

# class header_block(nn.Module):
#     def __init__(self, num_classes, beginchannel, channel, upornot=True):
#         super(header_block, self).__init__()
#         self.upornot = upornot
#         self.conv1 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
#         self.norm1 = nn.BatchNorm2d(channel)
#         self.conv2 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
#         self.norm2 = nn.BatchNorm2d(beginchannel)
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
#         self.conv3 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
#         self.norm3 = nn.BatchNorm2d(channel)
#         self.conv4 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
#         self.norm4 = nn.BatchNorm2d(beginchannel)
#         self.conv5 = nn.Conv2d(beginchannel, channel, 1, stride=1, bias=False)
#         self.norm5 = nn.BatchNorm2d(channel)

#         #yolo1,接conv5
#         self.conv7 = nn.Conv2d(channel, beginchannel, 3, stride=1, padding=1, bias=False)
#         self.norm7 = nn.BatchNorm2d(beginchannel)
#         self.conv8 = nn.Conv2d(beginchannel, (5+num_classes)*3, 1, stride=1, bias=False)
#         self.relu = nn.ReLU()

#         #upsample,接conv5
#         self.conv9 = nn.Conv2d(channel, channel//2, 1, stride=1, bias=False)
#         self.norm9 = nn.BatchNorm2d(channel//2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

#     def forward(self, x):
#         x = self.leaky_relu(self.norm1(self.conv1(x)))
#         x = self.leaky_relu(self.norm2(self.conv2(x)))
#         x = self.leaky_relu(self.norm3(self.conv3(x)))
#         x = self.leaky_relu(self.norm4(self.conv4(x)))
#         x = self.leaky_relu(self.norm5(self.conv5(x)))

#         y3 = self.leaky_relu(self.norm7(self.conv7(x)))
#         y3 = self.conv8(y3)
#         if self.upornot:
#             up3 = self.leaky_relu(self.norm9(self.conv9(x)))
#             up3 = self.upsample(up3)
#             return up3, y3
#         else:
#             return y3

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
#     union_area = boxes1_area + boxes2_area - inter_area + 1e-10
#     IOU = 1.0 * inter_area / union_area
#     enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
#     enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
#     enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
#     enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
#     # print(9999999999, IOU, enclose_area, union_area, enclose_area)
#     GIOU = IOU - 1.0 * (enclose_area - union_area) / enclose_area
#     return IOU

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
#     union_area = boxes1_area + boxes2_area - inter_area + 1e-10
#     IOU = 1.0 * inter_area / union_area
#     return IOU

# class yololayer(nn.Module):
#     '''
#     对输出的特征图进行解码，得到坐标、置信度、分类的概率
#     并计算相应的objectness loss、classify loss、coordinates loss
#     '''
#     def __init__(self, device, num_anchors, num_classes):
#         super(yololayer, self).__init__()
#         self.device = device
#         self.grid = torch.tensor([[],[]])
#         self.no = num_classes + 5
#         self.num_anchors = num_anchors

#     def forward(self, prediction, anchors, imgsize):
#         self.stride = imgsize//prediction.size(2)
#         batch_size, _, height, width = prediction.size() #batch_size, (5+num_classes)*3, width, height

#         #prediction [2, 75, 13, 13]
#         # prediction = prediction.view((batch_size, self.num_anchors, height, width, self.no)) #[2, 3, 13, 13, 25]
#         prediction = prediction.view(batch_size, self.num_anchors, self.no, height, width).permute(0, 1, 3, 4, 2).contiguous()
#         if not self.training:
#             # print(prediction.size(), self.grid[0].size())
#             if self.grid[0].size()[0] != prediction.size()[2]:
#                 self.grid = self._make_grid(prediction.size()[2], prediction.size()[2])
#             if prediction.device!=anchors.device:
#                 anchors = anchors.to('cuda')
#             # assert 1==0, anchors.device
#             prediction[..., 0] = (prediction[..., 0].sigmoid() + self.grid[0]) * self.stride #x #[2, 3, 13, 13]
#             prediction[..., 1] = (prediction[..., 1].sigmoid() + self.grid[1]) * self.stride #y #[2, 3, 13, 13]
#             prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchors #wh #[2, 3, 13, 13, 2]
#             # print(prediction.size(), self.grid[0].size())
#             # prediction[..., 0] = (prediction[..., 0].sigmoid() * 2. - 0.5 + self.grid[0]) * self.stride #x #[2, 3, 13, 13]
#             # prediction[..., 1] = (prediction[..., 1].sigmoid() * 2. - 0.5 + self.grid[1]) * self.stride #y #[2, 3, 13, 13]
#             # prediction[..., 2:4] = (prediction[..., 2:4] * 2) ** 2 * anchors #wh #[2, 3, 13, 13, 2]
#             prediction[..., 4:] = prediction[..., 4:].sigmoid()
#             prediction = prediction.view(batch_size, -1, self.no)
#         return prediction
            
#     def _make_grid(self, width, height):
#         x_coord = torch.arange(width).repeat(height, 1).unsqueeze(0).unsqueeze(0).to(self.device)    #[13, 13]
#         y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).unsqueeze(0).unsqueeze(0).to(self.device)     #[13, 13]
#         return [x_coord, y_coord]

# # This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py
# def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
#     # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
#     box2 = box2.T

#     # Get the coordinates of bounding boxes
#     if x1y1x2y2:  # x1, y1, x2, y2 = box1
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#     else:  # transform from xywh to xyxy
#         b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

#     # Intersection area
#     inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#             (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

#     # Union Area
#     w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#     w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
#     union = w1 * h1 + w2 * h2 - inter + eps

#     iou = inter / union
#     if GIoU or DIoU or CIoU:
#         # convex (smallest enclosing box) width
#         cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
#         ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#         if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#             c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#             rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
#                     (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
#             if DIoU:
#                 return iou - rho2 / c2  # DIoU
#             elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                 v = (4 / np.pi ** 2) * \
#                     torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#                 with torch.no_grad():
#                     alpha = v / ((1 + eps) - iou + v)
#                 return iou - (rho2 / c2 + v * alpha)  # CIoU
#         else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#             c_area = cw * ch + eps  # convex area
#             return iou - (c_area - union) / c_area  # GIoU
#     else:
#         return iou  # IoU

# def computeloss(predictions, targets, device, model):
#     # Check which device was used
#     device = targets.device

#     # Add placeholder varables for the different losses
#     lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

#     # Build yolo targets
#     tcls, tbox, indices, anchors = build_targets(predictions, targets, device, model)  # targets

#     # Define different loss functions classification
#     BCEcls = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([1.0], device=device))
#     BCEobj = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([1.0], device=device))

#     # Calculate losses for each yolo layer
#     for layer_index, layer_predictions in enumerate(predictions):
#         # Get image ids, anchors, grid index i and j for each target in the current yolo layer
#         b, anchor, grid_j, grid_i = indices[layer_index]
#         # Build empty object target tensor with the same shape as the object prediction
#         tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
#         # print(layer_predictions.size())
#         # exit(0)
#         # Get the number of targets for this layer.
#         # Each target is a label box with some scaling and the association of an anchor box.
#         # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
#         num_targets = b.shape[0]
#         # Check if there are targets for this batch
#         if num_targets:
#             # Load the corresponding values from the predictions for each of the targets
#             # print(b, anchor, grid_j, grid_i)
#             # print(layer_predictions.size())
#             # print(layer_predictions[b, anchor, grid_j, grid_i].size())
#             # exit(0)
#             ps = layer_predictions[b, anchor, grid_j, grid_i]

#             # Regression of the box
#             # Apply sigmoid to xy offset predictions in each cell that has a target
#             pxy = ps[:, :2].sigmoid()
#             # print(anchors.size())
#             # exit(0)
#             # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
#             pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
#             # Build box out of xy and wh
#             pbox = torch.cat((pxy, pwh), 1)
#             # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
#             iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
#             # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
#             lbox += (1.0 - iou).mean()  # iou loss

#             # Classification of the objectness
#             # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
#             tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets

#             # Classification of the class
#             # Check if we need to do a classification (number of classes > 1)
#             if ps.size(1) - 5 > 1:
#                 # Hot one class encoding
#                 t = torch.zeros_like(ps[:, 5:], device=device)  # targets
#                 t[range(num_targets), tcls[layer_index]] = 1
#                 # Use the tensor to calculate the BCE loss
#                 lcls += BCEcls(ps[:, 5:], t)  # BCE

#         # Classification of the objectness the sequel
#         # Calculate the BCE loss between the on the fly generated target and the network prediction
#         lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

#     lbox *= 0.05
#     lobj *= 1.0
#     lcls *= 0.5

#     # Merge losses
#     loss = lbox + lobj + lcls

#     return loss, torch.cat((lbox, lobj, lcls, loss)).cpu()

# def build_targets(p, targets, device, model):
#     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#     na, nt = 3, targets.shape[0]  # number of anchors, targets #TODO
#     tcls, tbox, indices, anch = [], [], [], []
#     gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#     # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
#     ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
#     # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
#     # print(targets)
#     # print(targets.repeat(na, 1, 1))
#     # print(ai[:, :, None])
#     # exit(0)
#     targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
#     # print(gain, ai, targets)
#     # exit(0)
#     for i in range(len(model.yolo)):
#         # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
#         stride = model.imgsize//p[0].size(2)
#         # assert 1==0, model.anchors[i]
#         anchors = torch.tensor(model.anchors[i], device=device) / stride
#         # print(yolo_layer.anchors, yolo_layer.stride)
#         # print(anchors)
#         # exit(0)
#         # Add the number of yolo cells in this layer the gain tensor
#         # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
#         gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#         # print(gain)
#         # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
#         t = targets * gain
#         # print(t)
#         # exit(0)
#         # Check if we have targets
#         if nt:
#             # Calculate ration between anchor and target box for both width and height
#             # print(anchors, anchors.size())
#             # exit(0)
#             r = t[:, :, 4:6] / anchors[:, None]
#             # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
#             j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO
#             # print(j.size())
#             # print(t.size())
#             # print(t[j])
#             # exit(0)
#             # Only use targets that have the correct ratios for their anchors
#             # That means we only keep ones that have a matching anchor and we loose the anchor dimension
#             # The anchor id is still saved in the 7th value of each target
#             t = t[j]
#         else:
#             t = targets[0]

#         # Extract image id in batch and class id
#         # print(t)
#         # print(t[:, :2].long())
#         # print(t[:, :2].long().T, t[:, :2].long().T.size())
#         b, c = t[:, :2].long().T
#         # print(b, c)
#         # exit(0)
#         # We isolate the target cell associations.
#         # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
#         gxy = t[:, 2:4]
#         gwh = t[:, 4:6]  # grid wh
#         # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
#         gij = gxy.long()
#         # Isolate x and y index dimensions
#         gi, gj = gij.T  # grid xy indices

#         # Convert anchor indexes to int
#         a = t[:, 6].long()
#         # Add target tensors for this yolo layer to the output lists
#         # Add to index list and limit index range to prevent out of bounds
#         indices.append((b, a, gj.clamp_(0, gain[3].long() - 1), gi.clamp_(0, gain[2].long() - 1)))
#         # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
#         tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#         # Add correct anchor for each target to the list
#         anch.append(anchors[a])
#         # Add class for each target to the list
#         tcls.append(c)
#     return tcls, tbox, indices, anch


# def build_targets(prediction, labels, device, model):
#     #生成labels, 是监督学习的，monitor learn，
#     # labels: [image_ind, class, ncx, ncy, nw, nh]
#     #prediction: (batch_size, num_anchors, height, width, 5 + num_classes)
#     na, nt = len(model.anchors[0]), labels.shape[0]  #每层anchor个数3，此batch图片的个数，number of anchors, targets
#     tcls, tbox, indices, anch = [], [], [], []
#     #每个labels都可能属于3个anchor对应输出层的任何一个层，所以labels要扩充3=na倍才可以，还要贴个标记，也就是在哪个anchor层
#     #each label may belong to any layer of three output layer to 3 anchor, so expand labels to triple. And add mark to which anchor layer
#     ai = torch.tensor([[i]*nt for i in range(na)], device=device) #[[0, 0], [1, 1], [2, 2]] nt=2, na=3
#     labels = torch.cat((labels.repeat(na, 1, 1), ai[:, :, None]), 2) #[na, nt, 7] (0img id, 1class, 2x, 3y, 4w, 5h, 6anchor id)
#     for i in range(len(prediction)): #yololayer output
#         copylabel = labels.clone()
#         #将anchor缩小到特征图大小，scale anchors by the yolo grid cell
#         stride = model.imgsize//prediction[i].size(2)     #16 416 prediction[i].size() torch.Size([2, 3, 26, 26, 7])
#         anchor = torch.tensor(model.anchors[i], device=device)/stride    #tensor([[0.7500, 1.3750], [2.2500, 3.6250], [4.4375, 8.5625]])
#         yolo_layer_width = prediction[i].size(3)
#         yolo_layer_height = prediction[i].size(2)
#         #将prediction的坐标从小数放大到特征图大小，scale labels by the yolo layer cell size
#         copylabel[..., [2, 4]] = copylabel[..., [2, 4]] * yolo_layer_width 
#         copylabel[..., [3, 5]] = copylabel[..., [3, 5]] * yolo_layer_height
#         #若labels的数量不是0 len(labels) > 0
#         if nt > 0:
#             #计算宽高比率， w, h ratio between anchor and label
#             r = copylabel[..., 4:6] / anchor[:, None]
#             #选取宽、高最大比率<4，不使用iou作判断，直接用宽高比率来决定labels用哪些anchor来Predict，一个label可以对应同层多个anchor
#             # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
#             #one label can predicted by multi anchor in this layer
#             j = torch.max(r, 1./r).max(2)[0] < 4
#             # Only use targets that have the correct ratios for their anchors
#             # That means we only keep ones that have a matching anchor and we loose the anchor dimension
#             # The anchor id is still saved in the 7th value of each target
#             # print(j.size(), copylabel.size())
#             copylabel = copylabel[j]
#         else:
#             copylabel = copylabel[0] #扩大了na倍，但是nt=0，不计算loss
#         #拿出image id和类别id, 并转置到行而不是列  long() -> 整数截断函数
#         # Extract image id in batch and class id
#         img_id, class_id = copylabel[..., :2].long().T
#         # We isolate the target cell associations.
#         #拿到放大到特征图大小的x,y,w,h坐标
#         # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
#         truth_xy = copylabel[..., 2:4]
#         truth_wh = copylabel[..., 4:6]
#         # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
#         # 拿到相应的ground_truth，x,y的网格坐标 long() -> 整数截断函数
#         truth_XY_long = truth_xy.long()
#         # Isolate x and y index dimensions
#         # 拿到截断以后网格坐标单独的x, y坐标
#         truth_X, truth_Y = truth_XY_long.T

#         # Convert anchor indexes to int
#         # anchor id到整数  long() -> 整数截断函数
#         anchor_id = copylabel[..., 6].long()
#         # Add target tensors for this yolo layer to the output lists
#         # Add to index list and limit index range to prevent out of bounds
#         indices.append((img_id, anchor_id, \
#             truth_Y.clamp_(0, yolo_layer_height - 1), \
#             truth_X.clamp_(0, yolo_layer_width - 1)))
#         # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
#         #x,y 的truth label
#         tbox.append(torch.cat((truth_xy - truth_XY_long, truth_wh), 1))
#         # Add correct anchor for each target to the list
#         #对应的anchor
#         anch.append(anchor[anchor_id])
#         # Add class for each target to the list
#         #对应的class_id
#         tcls.append(class_id)
#     return tcls, tbox, indices, anch

# def computeloss(prediction, labels, device, model):
#     #cls变量，box变量，置信度变量 add placeholder varables for different losses
#     lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

#     #build yolo targets 产生prediction对应的目标也就是labels，truth label
#     #是监督学习要相应的labels   monitor learning
#     truthcls, truthbox, indices, anchors = build_targets(prediction, labels, device, model)

#     #Define different loss functions classification
#     BCEcls = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([1.0], device=device))
#     BCEobj = nn.BCEWithLogitsLoss(
#         pos_weight=torch.tensor([1.0], device=device))
#     #对每层yololayer计算相应损失
#     # Calculate losses for each yolo layer
#     for layer_index, layer_predictions in enumerate(prediction):
#         # Get image ids, anchors, grid index i and j for each target in the current yolo layer
#         img_id, anchor_id, truth_Y, truth_X = indices[layer_index]
#         # Build empty object target tensor with the same shape as the object prediction
#         #产生placeholder，用来计算相应置信度
#         tobj = torch.zeros_like(layer_predictions[..., 0], device=device) #[2, 3, 26, 26, 7] [batchsize, num_anchor, gridheight, gridwidth, ...]
#         # Get the number of targets for this layer.
#         # Each target is a label box with some scaling and the association of an anchor box.
#         # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
#         num_labels = img_id.shape[0] #可能包含重复的，主要是可能多个anchor对应同一个label
#         # Check if there are targets for this batch
#         if num_labels > 0:
#             # Load the corresponding values from the predictions for each of the targets
#             #拿到对应的所有predict [..., 7]
#             predict = layer_predictions[img_id, anchor_id, truth_Y, truth_X]
#             # Regression of the box
#             # Apply sigmoid to xy offset predictions in each cell that has a target
#             # 应用sigmoid到xy的offset
#             predict_xy = predict[:, :2].sigmoid()
#             # predict_xy = predict[:, :2].sigmoid() * 2. - 0.5
#             # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
#             # 应用exp到wh
#             predict_wh = torch.exp(predict[:, 2:4]) * anchors[layer_index]
#             # predict_wh = (predict[:, 2:4].sigmoid() * 2) ** 2 * anchors[layer_index]
#             # Build box out of xy and wh
#             predict_boxes = torch.cat((predict_xy, predict_wh), 1)
#             # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
#             iou = bbox_iou(predict_boxes.T, truthbox[layer_index], x1y1x2y2=False, CIoU=True)
#             # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
#             lbox += (1.0 - iou).mean() #iouloss

#             # Classification of the objectness
#             # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
#             tobj[img_id, anchor_id, truth_Y, truth_X] = iou.detach().clamp(0).type(tobj.dtype) ## Use cells with iou > 0 as object targets
#             # Classification of the class
#             # Check if we need to do a classification (number of classes > 1)
#             if predict.size(1) - 5 > 1:
#                 # Hot one class encoding
#                 tkl = torch.zeros_like(predict[:, 5:], device=device)  # targets
#                 tkl[range(num_labels), truthcls[layer_index]] = 1
#                 # Use the tensor to calculate the BCE loss
#                 lcls += BCEcls(predict[:, 5:], tkl)    #BCE

#         # Classification of the objectness the sequel
#         # Calculate the BCE loss between the on the fly generated target and the network prediction
#         lobj += BCEobj(layer_predictions[..., 4], tobj)
#     lbox *= 0.05
#     lobj *= 1.0
#     lcls *= 0.05
#     # Merge losses
#     loss = lbox+lobj+lcls
#     res = torch.cat([lbox, lobj, lcls, loss])
#     return loss, res.detach().cpu()

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