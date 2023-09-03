#Author：ZouJiu
#Time: 2022-12-10
import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")

import sys
sys.path.append(abspath)
import torch
from datetime import datetime
from torchvision import transforms

# trainpath = r'/home/featurize/work/Pytorch_YOLOV3/2023/PyTorch-YOLOv3-master/data/person/personcartrain.txt'
trainpath = r'/root/autodl-tmp/annotations/person_keypoints_train2017.json'
train_imgpath = r'/root/autodl-tmp/train2017'
# train_imgpath = r'/root/autodl-tmp/train/images'
pth_evaluate = r'/root/autodl-tmp/annotations/person_keypoints_val2017.json'
img_evaluate = r'/root/autodl-tmp/val2017'
# img_evaluate = r'/root/autodl-tmp/val/images'
# img_evaluate = r'/root/autodl-tmp/val/images'

validpath = os.path.join(abspath, 'datas', 'cocoval.txt')
validsave = os.path.join(abspath, 'datas', 'cocoval', "predict")
validtruth = os.path.join(abspath, 'datas', 'cocoval', "truth")

# validpath = os.path.join(abspath, 'datas', 'val.txt')
# validsave = os.path.join(abspath, 'datas', 'val', "predict")
# validtruth = os.path.join(abspath, 'datas', 'val', "truth")
seed = 99999
#  CUDA_VISIBLE_DEVICES
#   /home/sh00259/.local/share/virtualenvs/Pytorch_YOLOV3-So1PrdKH/bin/python3.6 /home/sh00259/PycharmProjects/Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/allcodes/train_yolofastest.py
   
pretrainedmodel = r'/root/project/yolovkkn/2023-09-01yolokkn/model_e46_map[0.50716__0.334552]_lnan_2023-09-01.pt'
datekkk = datetime.strftime(datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')[:10]
logfile = os.path.join(abspath, 'log', 'log_yolov3_%s.txt'%datekkk)
darknet_weight = r"Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/pretrained"
# prefix = 'prunenight'
prefix = 'yolokkn'
savepath = r'/root/project/yolovkkn'
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# classes = []
# with open(os.path.join(abspath, 'datas', 'coconame2017.txt'), 'r', encoding='utf-8') as obj:
#     for i in obj.readlines():
#         i = i.strip()
#         classes.append(i)
# num_classes = 80

# assert num_classes==len(classes)

inputwidth = 32 * 16     # 32 * 16   32 * 19 
anchor_per_layer = 3
anchors = [[[10,13], [16,30], [33,23]],\
    [[30,61],  [62,45],  [59,119]],  \
    [[116,90],  [156,198],  [373,326]]]

ignore_thresh = 0.7 #iou>0.7 confidence loss
iou_thresh = 0.2
score_thresh = 0.38
nms_thresh = 0.5
strides = [8, 16, 32]

TFRESIZE = transforms.Compose([
    transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
])

TF = transforms.Compose([
    # transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

scratch = False
freeze_backbone = False
load_darknet_w = False
num_epochs = 100
warmepoch = 3
Adam = False
intialze = False

# [20230730, yolofive, Alexeydarknet, darknet, darknetRevise]
chooseLoss = "yolofive"
assert chooseLoss in ["20230730", "yolofive", "Alexeydarknet", "darknet", "darknetRevise"]

batch_size = 36
subsiz = 1
if Adam:
    learning_rate = 0.001                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
else:
    learning_rate = 0.01                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
momnetum=0.937
warmup_momnetum = 1.0 - 0.2
weight_decay= 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"