import torch
from datetime import datetime
from torchvision import transforms

trainpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train.txt'
validpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val.txt'
validsave = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val\predict'
validtruth = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\val\truth'

pretrainedmodel = r'./log/model_3_400_11.407_2022-07-10_10-09-54.pth'
tim = datetime.strftime(datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')
logfile = r'./log/log_%s.txt'%tim
darknet_weight = r"C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\log\yolov3.weights"

TF = transforms.Compose([
    transforms.Resize((416, 416)), 
    transforms.ToTensor(), #归一化到0到1之间
    # transforms.Normalize((0.5, 0.5, 0.5),(0.5,0.5,0.5)), #*-mean/std归一化到-1,1之间
])

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# num_classes = 20
classes = ['person', 'vehicle']
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

scratch = True
freeze_backbone = False
load_darknet_w = False
num_epochs = 139
Adam = True
intialze = False

batch_size = 18
learning_rate = 0.001 # initial learning rate (SGD=1E-2 0.01, Adam=1E-3 0.001)
momnetum=0.9
weight_decay= 0.0005

device = "cuda" if torch.cuda.is_available() else "cpu"




