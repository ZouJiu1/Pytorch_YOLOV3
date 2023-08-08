#Author：ZouJiu
#Time: 2022-12-10
import torch
from datetime import datetime
from torchvision import transforms

trainpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train.txt'
validpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val.txt'
validsave = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val\predict'
validtruth = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val\truth'
seed = 99999
#  CUDA_VISIBLE_DEVICES
#   /home/sh00259/.local/share/virtualenvs/Pytorch_YOLOV3-So1PrdKH/bin/python3.6 /home/sh00259/PycharmProjects/Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/allcodes/train_yolofastest.py

# pretrainedmodel = r'/Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/log/model_100_3333_map0.7492814498144875_0.830_2022-11-12_18-19-40.pth'
pretrainedmodel = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\log\yolofastestrevise\2023-03-05yflight\model_9_29670_map0_0.226_2023-03-05.pth'
tim = datetime.strftime(datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')[:10]
logfile = r'./log/log_%s.txt'%tim
darknet_weight = r"Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/pretrained/xx.il"
# prefix = 'prunenight'
prefix = 'yflight'
savepath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\log\yolofastestrevise'
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# num_classes = 20
classes = ['person', 'car']
num_classes = 2

assert num_classes==len(classes)

inputwidth = 416
anchor_per_layer = 3
# anchors = [[[10,13], [16,30], [33,23]],\
#     [[30,61],  [62,45],  [59,119]],  \
#     [[116,90],  [156,198],  [373,326]]]
anchors = [[[12,22], [36,58], [71,137]], \
            [[126,271], [215,150], [305,350]]]
ignore_thresh = 0.5 #iou>0.7 confidence loss
iou_thresh = 0.5
score_thresh = 0.38
nms_thresh = 0.5
strides = [16, 32] #[8, 16, 32]

TF = transforms.Compose([
    transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

scratch = False
freeze_backbone = False
load_darknet_w = False
num_epochs = 139
Adam = True
intialze = False

batch_size = 2
if Adam:
    learning_rate = 0.001                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
else:
    learning_rate = 0.006                                      #initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
momnetum=0.9
weight_decay= 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"