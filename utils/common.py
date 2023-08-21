import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import cv2
import torch
import numpy as np
from PIL import Image
import math
from torch.utils.data import Dataset
from config.config_yolov3tiny import inputwidth, classes
import random
from copy import deepcopy
from torch import nn
import albumentations as A



counts=0
def cvshow_(image, label):
    import cv2
    import numpy as np
    global counts
    savepath = os.path.join(abspath, 'datas', 'imshow')
    for i in os.listdir(savepath):
        os.remove(os.path.join(savepath, i))
    lis = [[] for ij in range(image.size()[0])]
    # classes = ['person', 'car']
    for j in range(label.size()[0]):
        lab = int(label[j][0])
        cla = int(label[j][1])
        xmin = inputwidth * (label[j][2] - label[j][4]/2)
        ymin = inputwidth * (label[j][3] - label[j][5]/2)
        xmax = inputwidth * (label[j][2] + label[j][4]/2)
        ymax = inputwidth * (label[j][3] + label[j][5]/2)
        lis[lab].append([cla, int(xmin), int(ymin), int(xmax), int(ymax)])

    font = cv2.FONT_HERSHEY_SIMPLEX
    for j in range(image.size()[0]):
        img = image[j].detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = np.array(img*255, dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for ij in lis[j]:
            print(ij, img.shape)
            cv2.rectangle(img, (ij[1], ij[2]), (ij[3], ij[4]), [0, 0, 128], 1)
            cv2.putText(img, classes[ij[0]], (ij[1], ij[2]),
                        font, 0.6, (128, 0, 128), 1)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(savepath, str(counts) + '.jpg'), img)
        counts += 1


counts = 0

def cvshow(image, label):
    global counts
    savepath = r'datas\imshow'
    lis = [[] for ij in range(image.size()[0])]
    for j in range(label.size()[0]):
        lab = int(label[j][-1])
        cla = int(label[j][0])
        xmin = inputwidth * (label[j][1] - label[j][3]/2)
        ymin = inputwidth * (label[j][2] - label[j][4]/2)
        xmax = inputwidth * (label[j][1] + label[j][3]/2)
        ymax = inputwidth * (label[j][2] + label[j][4]/2)
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

def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    from torch import nn
    #https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#L318
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        for p_name, p in v.named_parameters(recurse=0):
            if p_name == 'bias':  # bias (no decay)
                g[2].append(p)
            elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                g[1].append(p)
            else:
                g[0].append(p)  # weight (with decay)

    if name == 'Adam':
        optimizer = torch.optim.Adam(g[2], lr=lr, betas=(momentum, 0.999))  # adjust beta1 to momentum
    elif name == 'AdamW':
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    elif name == 'RMSProp':
        optimizer = torch.optim.RMSprop(g[2], lr=lr, momentum=momentum)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    else:
        raise NotImplementedError(f'Optimizer {name} not implemented.')

    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})  # add g0 with weight_decay
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer

def de_parallel(model):
    if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model
    
def copy_attr(a, b, include=(), exclude=()):
    # https://github.com/ultralytics/yolov5/
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)

class ModelEMA:
    # https://github.com/ultralytics/yolov5/
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)

def collate_fn(batch):
    batch = [i for i in batch if i is not None]
    # print(batch)
    img, label, image_id = list(zip(*batch))  # transposed
    # print(len(label))
    for i, l in enumerate(label):
        l[:, 0] = i
    images = torch.stack(img)
    labels = torch.cat(label, 0)
    return images, labels, image_id

def collate_fn_tails(batch):
    batch = [i for i in batch if i is not None]
    img, label = list(zip(*batch))  # transposed
    # print(len(label))
    for i, l in enumerate(label):
        l[:, -1] = i
    images = torch.stack(img)
    labels = torch.cat(label, 0)
    return images, labels

def provide_determinism(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ia.seed(seed)

    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

class validDataset(Dataset):
    def __init__(self, traintxt, transform=None):
        super(validDataset, self).__init__()
        self.transform = transform
        self.trainpath = []
        self.traintxt = traintxt
        with open(self.traintxt, 'r') as f:
            for i in f.readlines():
                i = i.strip()
                self.trainpath.append(i)

    def __len__(self):
        return len(self.trainpath)
    
    def __getitem__(self, idx):
        imgpath = self.trainpath[idx]
        image = cv2.imread(imgpath)
        nam = imgpath.split(os.sep)[-1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, nam