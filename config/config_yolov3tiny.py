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

# trainpath = r'/root/project/yolov3tiny/Pytorch_YOLOV3/2023/PyTorch-YOLOv3-master/data/person/personcartrain.txt'
trainpath = r'/root/autodl-tmp/annotations/instances_train2017.json'
train_imgpath = r'/root/autodl-tmp/train2017'
pth_evaluate = r'/root/autodl-tmp/annotations/instances_val2017.json'
img_evaluate = r'/root/autodl-tmp/val2017'

validpath = os.path.join(abspath, 'datas', 'cocoval.txt')
validsave = os.path.join(abspath, 'datas', 'cocoval', "predict")
validtruth = os.path.join(abspath, 'datas', 'cocoval', "truth")

# validpath = os.path.join(abspath, 'datas', 'val.txt')
# validsave = os.path.join(abspath, 'datas', 'val', "predict")
# validtruth = os.path.join(abspath, 'datas', 'val', "truth")

seed = 99999
#  CUDA_VISIBLE_DEVICES
#   /home/sh00259/.local/share/virtualenvs/Pytorch_YOLOV3-So1PrdKH/bin/python3.6 /home/sh00259/PycharmProjects/Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/allcodes/train_yolofastest.py

pretrainedmodel = r'/root/project/yolov3tiny/2023-08-08yolov3tiny_Alexay_mse/model_e71_map[0.30188__0.099243]_l493.264_2023-08-08.pth'
# pretrainedmodel = r'/root/project/yolov3tiny/yolov3tiny.pth'
# pretrainedmodel = r'/root/project/yolov3tiny/model_e114_t334248_map[0.21699288869531463,_0.003444264162438733]_iou_scale_anchorandth_iou_before_noclassdiff_l0.017_2023-07-28.pth'
datekkk = datetime.strftime(datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')[:10]
logfile = os.path.join(abspath, 'log', 'log_yolov3tiny_%s.txt'%datekkk)
darknet_weight = r"Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/pretrained"
# prefix = 'prunenight'
prefix = 'yolov3tiny_yolofive_0.01_0.937_bce_ncd'
savepath = r'/root/project/yolov3tiny'
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = []
with open(os.path.join(abspath, 'datas', 'coconame2017.txt'), 'r', encoding='utf-8') as obj:
    for i in obj.readlines():
        i = i.strip()
        classes.append(i)
num_classes = 80

assert num_classes==len(classes)

inputwidth = 32 * 16     # 32 * 16   32 * 19 
anchor_per_layer = 3
# anchors = [[[10,13], [16,30], [33,23]],\
#     [[30,61],  [62,45],  [59,119]],  \
#     [[116,90],  [156,198],  [373,326]]]
anchors = [[[10,13], [23,27], [37,58]], \
            [[81,82], [135,169], [339,319]]]
ignore_thresh = 0.7 #iou>0.7 confidence loss
iou_thresh = 0.2
score_thresh = 0.38
nms_thresh = 0.5
strides = [16, 32] #[8, 16, 32]

TFRESIZE = transforms.Compose([
    transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
])

TF = transforms.Compose([
    # transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

scratch = True
freeze_backbone = False
load_darknet_w = False
num_epochs = 69
warmepoch = 3
Adam = False
intialze = False

multi_GPU = True
darknetReviseLoss = False
darknetLoss = True
AlexeydarknetLoss = False
yolofiveeightLoss = False

assert (darknetReviseLoss + darknetLoss + AlexeydarknetLoss + yolofiveeightLoss) == 1

batch_size = 36
subsiz = 1
# if Adam:
    # learning_rate = 0.001                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
# else:
learning_rate = 0.001                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
momnetum=0.937
warmup_momnetum = 1.0 - 0.2
weight_decay= 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"