#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import time
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.append(r'/home/Pytorch_YOLOV3')
from loaddata.load_datas_731 import TF, trainDataset, collate_fn
from models.Yolov3_731 import *
import torch.optim as optim


# path = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\JPEGImages'
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["person", "bird"]
# counts = 0 
def cvshow(image, label):
    global counts
    savepath = r'datas\imshow'
    lis = [[] for ij in range(image.size()[0])]
    for j in range(label.size()[0]):
        lab = int(label[j][-1])
        cla = int(label[j][0])
        xmin = 416 * (label[j][1] - label[j][3]/2)
        ymin = 416 * (label[j][2] - label[j][4]/2)
        xmax = 416 * (label[j][1] + label[j][3]/2)
        ymax = 416 * (label[j][2] + label[j][4]/2)
        lis[lab].append([cla, int(xmin), int(ymin), int(xmax), int(ymax)])

    font = cv2.FONT_HERSHEY_SIMPLEX
    for j in range(image.size()[0]):
        img = image[j].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.array(img*255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for ij in lis[j]:
            # print(ij, img.shape, ij[1], ij[2], ij[3], ij[4])
            cv2.rectangle(img, (ij[1], ij[2]), (ij[3], ij[4]), [0, 0, 128], 1)
            cv2.putText(img, classes[ij[0]], (ij[1], ij[2]),
                        font, 0.6, (128, 0, 128), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(savepath, str(counts) + '.jpg'), img)
        counts += 1

device = "cuda" if torch.cuda.is_available() else "cpu"

def intialization(model):
    '''https://github.com/Peterisfar/YOLOV3/blob/master/model/yolov3.py#L68'''
    for name, m in model.named_children(): #1 
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
            print("initialized: {}".format(m))
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight.data, 1.0)
            torch.nn.init.constant_(m.bias.data, 0.0)
            print("initialized: {}".format(m))
        else:
            intialization(m)

def loadweight(weights, count, m, ptr):
    # if count == cutoff:
    #     return None
    # count += 1
    layers = []
    try:
        norm1 = m.norm1
        layers.append(norm1)
    except:
        pass
    try:
        conv1 = m.conv1
        layers.append(conv1)
    except:
        pass
    try:
        norm2 = m.norm2
        layers.append(norm2)
    except:
        pass
    try:
        conv2 = m.conv2
        layers.append(conv2)
    except:
        pass
    try:
        norm_extra = m.norm_extra
        layers.append(norm_extra)
    except:
        pass
    try:
        conv_extra = m.conv_extra
        layers.append(conv_extra)
    except:
        pass
    for mn in layers:
        conv_layer = mn
        if isinstance(mn, nn.BatchNorm2d):
            # Load BN bias, weights, running mean and running variance
            bn_layer = mn
            num_b = bn_layer.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
            bn_layer.bias.data.copy_(bn_b)
            ptr += num_b
            # Weight
            bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
            bn_layer.weight.data.copy_(bn_w)
            ptr += num_b
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
            bn_layer.running_mean.data.copy_(bn_rm)
            ptr += num_b
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
            bn_layer.running_var.data.copy_(bn_rv)
            ptr += num_b

            print("loaded {}".format(bn_layer))
        else:
            # Load conv. bias
            if conv_layer.bias!=None:
                num_b = conv_layer.bias.numel()
                conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                conv_layer.bias.data.copy_(conv_b)
                ptr += num_b
            # Load conv. weights
            num_w = conv_layer.weight.numel()
            conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
            conv_layer.weight.data.copy_(conv_w)
            ptr += num_w
            print("loaded {}".format(conv_layer))

    return count, ptr

def load_darknet_weights(model: torch.nn.Module, weight_file, cutoff=52):
    "https://github.com/ultralytics/yolov3/blob/master/models.py"

    print("load darknet weights : ", weight_file)

    with open(weight_file, 'rb') as f:
        _ = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)
    count = 0
    ptr = 0

    for name, m in model.named_children(): #1 
        if isinstance(m, inputnet):
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock1):
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock2):
            for i in range(len(m.resblock2_child)):
                count, ptr = loadweight(weights, count, m.resblock2_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock3):
            for i in range(len(m.resblock3_child)):
                count, ptr = loadweight(weights, count, m.resblock3_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock4):
            for i in range(len(m.resblock4_child)):
                count, ptr = loadweight(weights, count, m.resblock4_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)
        elif isinstance(m, resblock5):
            for i in range(len(m.resblock5_child)):
                count, ptr = loadweight(weights, count, m.resblock5_child[i], ptr)
            count, ptr = loadweight(weights, count, m, ptr)

def freeze_darknet_backbone(model):
    '''https://github.com/qqwweee/keras-yolo3/blob/master/train.py#L50'''
    freeze_netlist = ["net", "ResBlock1", "ResBlock2", "ResBlock3", "ResBlock4", "ResBlock5"]
    for name, m in model.named_parameters(): #1 
        mar = 999
        for i in freeze_netlist:
            if i in name:
                mar = -999
                break
        if mar < 0:
            print("freeze layer params: ", name, list(m.size()))
            m.requires_grad = False
        else:
            print("train layer params: ", name, list(m.size()))
            continue

def evaluate(model, dataloader_test, device = device):
    valdata = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\val.txt'
    traindata = trainDataset(valdata, transform=TF)
    dataloader = DataLoader(traindata, batch_size=1,shuffle=False, \
        num_workers=2,collate_fn=collate_fn)
    model.eval()
    for i, (image, label) in enumerate(dataloader):
        # optimizer.zero_grad()
        image = image.to(device)
        label = label.to(device)
        y3, y2, y1, result3, result2, result1 = model(image, '')
        print(result3)
        print(result2)
        print(result1)
        result = result3 + result2 + result
        break