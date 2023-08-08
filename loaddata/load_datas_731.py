#Author：ZouJiu
#Time: 2021-8-13

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from models.Yolov3_731 import box_iou
import albumentations as A

# Declare an augmentation pipeline
P = 0.03
AAAtransform = A.Compose([
    A.HorizontalFlip(p=P),
    A.RandomGamma(p=P),
    A.HueSaturationValue(p=P),
    A.RandomBrightnessContrast(p=P),
    A.MotionBlur(p=P),
    A.GaussianBlur(p=P),
    A.GaussNoise(p=P),
    A.ToGray(p=P),
    A.Equalize(p=P),
    A.PixelDropout(p=0.003),
    A.RandomBrightness(p=P)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

class trainDataset(Dataset):
    def __init__(self, traintxt, stride, anchors, anchor_per_layer, \
                device, inputwidth, numclasses,\
                transform=None, target_transform=None):
        super(trainDataset, self).__init__()
        self.traintxt = traintxt
        self.transform = transform
        self.target_transform = target_transform
        self.trainpath = []
        self.stride = np.array(stride)  #[8, 16, 32]
        self.anchors = np.array(anchors)
        self.inputwidth = inputwidth
        self.numclasses = numclasses
        self.device = device
        self.anchor_per_layer = anchor_per_layer
        with open(self.traintxt, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                self.trainpath.append(i)

        maxins = -999999
        for i in self.trainpath:
            cnt = 0
            txt = i.replace('JPEGImages','labels').split('.')[0]+'.txt'
            with open(txt, 'r') as obj:
                for j in obj.readlines():
                    cnt += 1
            if cnt>maxins:
                maxins = cnt
        self.maxrec = maxins
        # print(1111111111111111, len(self.trainpath))

    def __len__(self):
        return len(self.trainpath)
    
    def __getitem__(self, idx):
        imgpath = self.trainpath[idx]
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', image.detach().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        labelpath = imgpath.replace('JPEGImages','labels').split('.')[0]+'.txt'
        bboxes = []
        labels = []
        with open(labelpath, 'r') as f:
            for i in f.readlines():
                label, cx, cy, w, h = i.strip().split(' ')
                # gt.append([int(label), float(cx), float(cy), float(w), float(h), 0])
                bboxes.append([float(cx), float(cy), float(w), float(h)])
                labels.append(int(label))
        bboxes = np.array(bboxes)
        if self.target_transform:
            bboxes = self.target_transform(bboxes)
        transformed = AAAtransform(image=image, bboxes = bboxes, class_labels = labels)
        image = transformed['image']
        gtt = transformed['bboxes']
        class_labels = transformed['class_labels']
        gt = []
        for ind, i in enumerate(gtt):
            gt.append([class_labels[ind], i[0], i[1], i[2], i[3], 0])
        gt = np.array(gt)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        
        label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox = self.__create_labels(gt)
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()

        sbbox = torch.from_numpy(sbbox).float()
        mbbox = torch.from_numpy(mbbox).float()
        lbbox = torch.from_numpy(lbbox).float()
        return image, label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox
    
    def __create_labels(self, bboxes):
        train_outsize = self.inputwidth/self.stride #[52, 26, 13]
        labels = [ np.zeros((int(train_outsize[i]), int(train_outsize[i]), \
            self.anchor_per_layer, 5 + self.numclasses)) for i in range(3)]
        
        max_rec= self.maxrec #提前配置的单个图片最多框的个数
        bboxes_txywh = [np.zeros((max_rec, 4)) for _ in range(3)]
        bboxes_count = np.zeros((3,))

        for i in range(len(bboxes)):
            rectangle = bboxes[i]
            cls = int(rectangle[0])
            ncx, ncy, nw, nh = rectangle[1], rectangle[2], rectangle[3], rectangle[4]
            iou = []
            exist_positive = False
            for ja in range(3):              #small boxes、middle boxes、large boxes
                cx = int(np.floor(ncx * train_outsize[ja]))
                cy = int(np.floor(ncy * train_outsize[ja]))
                #由于anchor是相对原图的，所以这里都放到原图尺寸，然后计算相应IOU
                gtone = torch.zeros((1, 3 + 1))  #xmin  ymin  xmax  ymax
                gtone[:, 0] = (ncx - nw / 2) * self.inputwidth
                gtone[:, 1] = (ncy - nh / 2) * self.inputwidth
                gtone[:, 2] = (ncx + nw / 2) * self.inputwidth
                gtone[:, 3] = (ncy + nh / 2) * self.inputwidth

                anchors = self.anchors[ja]
                anchorxyxy = torch.zeros((self.anchor_per_layer, 3+1))
                anchorxyxy[:, 0] = torch.from_numpy(ncx * self.inputwidth - anchors[:, 0] / 2)   #原图尺寸的坐标
                anchorxyxy[:, 1] = torch.from_numpy(ncy * self.inputwidth - anchors[:, 1] / 2)
                anchorxyxy[:, 2] = torch.from_numpy(ncx * self.inputwidth + anchors[:, 0] / 2)
                anchorxyxy[:, 3] = torch.from_numpy(ncy * self.inputwidth + anchors[:, 1] / 2)
                one_iou = box_iou(gtone, anchorxyxy)[0] #[1, 3]
                iou.extend(one_iou.detach().cpu().numpy())
                ioumask = one_iou > 0.3

                if(torch.sum(ioumask).item() > 0):
                    labels[ja][cy, cx, ioumask, 0] = ncx * self.inputwidth   #center x     #原图尺寸的坐标
                    labels[ja][cy, cx, ioumask, 1] = ncy * self.inputwidth  #center y
                    labels[ja][cy, cx, ioumask, 2] = nw * self.inputwidth  #rectangle w
                    labels[ja][cy, cx, ioumask, 3] = nh * self.inputwidth  #rectangle h
                    labels[ja][cy, cx, ioumask, 4] = 1  #confidence
                    labels[ja][cy, cx, ioumask, 5 + cls] = 1  #classify class
                    exist_positive = True

                    bboxes_ind = int(bboxes_count[ja] % max_rec)
                    bboxes_txywh[ja][bboxes_ind, :] = [ncx * self.inputwidth, ncy * self.inputwidth,\
                                                   nw * self.inputwidth, nh * self.inputwidth]    #原图尺寸的坐标
                    bboxes_count[ja] += 1

            #不存在相应的预测
            if not exist_positive:
                maxind = np.argmax(np.array(iou).reshape(-1))
                layer = int(maxind/3)    #[52, 26, 13]  which layer belong
                layer_a = int(maxind%3)       # which anchor of 3
                labels[layer][cy, cx, layer_a, 0] = ncx * self.inputwidth   #center x
                labels[layer][cy, cx, layer_a, 1] = ncy * self.inputwidth  #center y
                labels[layer][cy, cx, layer_a, 2] = nw * self.inputwidth  #rectangle w
                labels[layer][cy, cx, layer_a, 3] = nh * self.inputwidth  #rectangle h
                labels[layer][cy, cx, layer_a, 4] = 1  #confidence 
                labels[layer][cy, cx, layer_a, 5 + cls] = 1  #classify class

                bboxes_ind = int(bboxes_count[layer] % max_rec)
                bboxes_txywh[layer][bboxes_ind, :] = [ncx * self.inputwidth, ncy * self.inputwidth,\
                                                   nw * self.inputwidth, nh * self.inputwidth]    #原图尺寸的坐标
                bboxes_count[layer] += 1

        label_lbbox, label_mbbox, label_sbbox = labels
        lbbox, mbbox, sbbox = bboxes_txywh
        return label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox

if __name__ == '__main__':
    trainpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\train.txt'
    inputwidth = 416
    anchors = [[[10,13], [16,30], [33,23]],\
        [[30,61],  [62,45],  [59,119]],  \
        [[116,90],  [156,198],  [373,326]]]
    strides = [8, 16, 32]

    # anchors = np.array(anchors, dtype = np.float32)
    # print(anchors.shape)
    # for i in range(3):
    #     anchors[i, ...] = anchors[i, ...]/strides[i]
    # print(anchors)
    # exit(0)

    anchor_per_layer = 3
    num_classes = 2 #voc2007_2012 20
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traindata = trainDataset(trainpath, stride = strides, anchors = anchors, anchor_per_layer = anchor_per_layer, \
                             device = device, inputwidth = inputwidth, numclasses = num_classes,transform=TF)
    for i, (images, labels) in enumerate(traindata):
        print(images.size(), labels.size())