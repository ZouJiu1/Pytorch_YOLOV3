#Author：ZouJiu
#Time: 2022-12-10

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

# Declare an augmentation pipeline
P = 0.03
AAAtransform = A.Compose([
    # A.HorizontalFlip(p=P),
    A.RandomGamma(p=P),
    # A.HueSaturationValue(p=P),
    A.RandomBrightnessContrast(p=P),
    # A.MotionBlur(p=P),
    A.GaussianBlur(p=P),
    # A.GaussNoise(p=P),
    A.ToGray(p=P),
    A.Equalize(p=P),
    # A.PixelDropout(p=0.003),
    # A.RandomBrightness(p=P)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

class trainDataset(Dataset):
    def __init__(self, traintxt, stride, anchors, inputwidth, augment=True, transform=None, target_transform=None):
        super(trainDataset, self).__init__()
        self.traintxt = traintxt
        self.transform = transform
        self.target_transform = target_transform
        self.trainpath = []
        self.stride = stride
        self.anchors = anchors
        self.inputwidth = inputwidth
        self.augment = augment
        with open(self.traintxt, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                self.trainpath.append(i)

    def __len__(self):
        return len(self.trainpath)
    
    def __getitem__(self, idx):
        inpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\checkpoints'
        imgpath = self.trainpath[idx]
        image = cv2.imread(imgpath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        labelpath = imgpath.replace('JPEGImages','labels').replace('images', 'labels').replace(".jpg", '.txt')
        bboxes = []
        labels = []
        gt = []
        # print(labelpath, imgpath)
        with open(labelpath, 'r') as f:
            for i in f.readlines():
                label, cx, cy, w, h = i.strip().split(' ')
                # if int(label) not in [0, 2]:
                #     continue
                # if int(label)==2:
                #     label=1
                label, cx, cy, w, h = int(label), float(cx), float(cy), float(w), float(h)
                    
                gt.append([0, int(label), float(cx), float(cy), float(w), float(h)])
                bboxes.append([float(cx), float(cy), float(w), float(h)])
                labels.append(int(label))
        bboxes = np.array(bboxes)
        # print(bboxes)
        if self.target_transform:
            bboxes = self.target_transform(bboxes)
        if self.augment:
            try:
                transformed = AAAtransform(image=image, bboxes = bboxes, class_labels = labels, )
                image = transformed['image']
                gtt = transformed['bboxes']
                class_labels = transformed['class_labels']
                kkk = []
                for ind, i in enumerate(gtt):
                    kkk.append([0, class_labels[ind], i[0], i[1], i[2], i[3]])
                gt = np.array(kkk)
            except Exception as e:
                print("e")
                gt = np.array(gt)
                pass
        else:
            gt = np.array(gt)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(gt)

if __name__ == '__main__':
    trainpath = r'/home/Pytorch_YOLOV3\datas\train\train.txt'
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
