import torch
from datetime import datetime
from torchvision import transforms

trainpath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\datas\train\train.txt'

validpath = r'datas/valid/valid.txt'
validsave = r'datas/valid/predict'
validtruth = r'datas/valid/labels'

pretrainedmodel = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\log\730\model_80_34101_map0.0002314929361444335_3.018_2022-09-10_07-38-56.pth'
tim = datetime.strftime(datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')
logfile = r'./log/log_%s.txt'%tim
darknet_weight = r"C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\log\yolov3.weights"
savepath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\log\730'
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ['person', 'car']
num_classes = 2

assert num_classes==len(classes)

inputwidth = 416
anchor_per_layer = 3
anchors = [[[10,13], [16,30], [33,23]],\
    [[30,61],  [62,45],  [59,119]],  \
    [[116,90],  [156,198],  [373,326]]]
ignore_thresh = 0.5 #iou>0.7 confidence loss
iou_thresh = 0.5
score_thresh = 0.38
nms_thresh = 0.5
strides = [8, 16, 32]

TF = transforms.Compose([
    transforms.Resize((inputwidth, inputwidth)), 
    transforms.ToTensor(), #归一化到0到1之间
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

scratch = True
freeze_backbone = False
load_darknet_w = False
num_epochs = 139
Adam = True
intialze = False

batch_size = 8
learning_rate = 0.0001 # initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
momnetum=0.9
weight_decay= 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"




