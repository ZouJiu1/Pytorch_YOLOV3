#Author：ZouJiu
#Time: 2021-8-13

import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TF = transforms.Compose([
    transforms.Resize((416, 416)), 
    transforms.ToTensor(), #归一化到0到1之间
    transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        try:
            #print(111, l.size())
            l[:, -1] = i  # add target image index for build_targets()
        except Exception as e:
            l = torch.Tensor([[0,0.5,0.1,0.1,0.1,i]])
            print(999, l.size())
    #print(333, torch.cat(label, 0).size(), e) 
    return torch.stack(img, 0), torch.cat(label, 0)

class trainDataset(Dataset):
    def __init__(self, traintxt, transform=None, target_transform=None):
        super(trainDataset, self).__init__()
        self.traintxt = traintxt
        self.transform = transform
        self.target_transform = target_transform
        self.trainpath = []
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
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        # cv2.imshow('img', image.detach().cpu().numpy())
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        labelpath = imgpath.replace('JPEGImages','labels').split('.')[0]+'.txt'
        gt = []
        sample = {}
        with open(labelpath, 'r') as f:
            for i in f.readlines():
                label, cx, cy, w, h = i.strip().split(' ')
                gt.append([int(label), float(cx), float(cy), float(w), float(h), 0])
        if self.target_transform:
            gt = self.target_transform(gt)
        return image, torch.tensor(gt)
