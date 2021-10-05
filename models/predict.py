#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import time  
import cv2
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from load_datas import TF, trainDataset, collate_fn
from Yolov3 import Yolov3
from PIL import Image
import torch.optim as optim

pretrainedmodel = r'C:\Users\10696\Desktop\yolov3\log\model_337_881000_0.001_2021-09-03_20-36-52.pth'
imgpath = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\test.txt'
# imgpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3myself\cocoval2017.txt'
device = "cuda" if torch.cuda.is_available() else "cpu"
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
num_classes = 20
inputwidth = 416 
anchors = [[10,13], [16,30], [33,23],\
    [30,61],  [62,45],  [59,119],  \
    [116,90],  [156,198],  [373,326]]
ignore_thresh = 0.5 #iou>0.7 confidence loss
score_thresh = 0.1
nms_thresh = 0.5
model = Yolov3(num_classes, anchors, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
if torch.cuda.is_available():
    state_dict = torch.load(pretrainedmodel,map_location=torch.device('cuda')) 
else:
    state_dict = torch.load(pretrainedmodel,map_location=torch.device('cpu')) 
model.load_state_dict(state_dict['state_dict'])
print('loaded', pretrainedmodel)
model = model.to(device)

lis = []
with open(imgpath, 'r') as f:
    for i in f.readlines():
        i = i.strip()
        lis.append(i)

TF = transforms.Compose([
    transforms.Resize((416, 416)), 
    transforms.ToTensor(), #归一化到0到1之间
    transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])
device = 'cuda' if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
# np.random.shuffle(lis)
for ind, i in enumerate(lis):
    print(ind, i)
    # img = cv2.imread(i)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # h,w,c=img.shape
    # img = cv2.resize(image, (416, 416))
    # img = img/255
    # img = (img-0.5)/0.5
    # img = np.expand_dims(img, 0)
    # img = np.transpose(img, (0, 3, 1, 2))
    # img = torch.FloatTensor(img)

    image = Image.open(i).convert("RGB")
    w, h = image.size
    img = TF(image)
    img = torch.unsqueeze(img, 0).to(device)
    p, _,  _,  _,  _,  _, _, _= model(img)
    if p.shape[1]==0:
        continue
    p = p[0]
    cxp, cyp, wp, hp, maxscore, label = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]
    xmin = (cxp - wp/2)*w
    ymin = (cyp - hp/2)*h
    xmax = (cxp + wp/2)*w
    ymax = (cyp + hp/2)*h
    cvfont = cv2.FONT_HERSHEY_SIMPLEX
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for j in range(len(label)):
        minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin[j]))), min(h-1, max(0, int(ymin[j]))), \
            min(w-1, max(0, int(xmax[j]))), min(h-1, max(0, int(ymax[j])))
        try:
            cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 1)
        except Exception as e:
            print(e)
            continue
        text = classes[int(label[j])] + ' ' + str(round(maxscore[j],3))
        # text = text.replace('4','9')
        cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 255], 1)
        # print(classes[int(label[j])], end=' ')
    # print() 
    na = i.split(os.sep)[-1]
    cv2.imwrite(r'images\%s'%na, image)


    # image = cv2.resize(image, (300,600))
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


