#Author：ZouJiu
#Time: 2021-8-13

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A

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
    def __init__(self, traintxt, stride, anchors, inputwidth, transform=None, target_transform=None):
        super(trainDataset, self).__init__()
        self.traintxt = traintxt
        self.transform = transform
        self.target_transform = target_transform
        self.trainpath = []
        self.stride = stride
        self.anchors = anchors
        self.inputwidth = inputwidth
        with open(self.traintxt, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                self.trainpath.append(i)
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
        
        return image, torch.tensor(gt)
