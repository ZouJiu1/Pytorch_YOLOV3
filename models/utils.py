#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import time
import cv2
from torch.utils.data import Dataset, DataLoader
from load_datas import TF, trainDataset, collate_fn
from Yolov3 import Yolov3
import torch.optim as optim

path = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\JPEGImages'

def evaluate(model, dataloader_test, device = device):
    valdata = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\val.txt'
    traindata = trainDataset(valdata, transform=TF)
    dataloader = DataLoader(traindata, batch_size=1,shuffle=False, \
        num_workers=2,collate_fn=collate_fn)
    model.eval()
    for i, (image, label) in enumerate(dataloader):
        optimizer.zero_grad()
        image = image.to(device)
        label = label.to(device)
        y3, y2, y1, result3, result2, result1 = model(image, '')
        print(result3)
        print(result2)
        print(result1)
        result = result3 + result2 + result
        break