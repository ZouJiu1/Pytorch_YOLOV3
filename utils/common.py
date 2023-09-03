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



counts=0
def cvshow_seg(image, label, masks):
    import cv2
    import numpy as np
    global counts
    savepath = os.path.join(abspath, 'datas', 'seg')
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
        for ij in range(len(masks)):
            kk = label[:, 0].long()==j
            if not kk[ij]:
                continue
            ij = masks[ij].cpu().numpy()
            ijb = ij * 200
            ij = np.stack([ijb, ij, ij])
            ij = np.transpose(ij, (1, 2, 0))
            
            # cv2.imshow('img', ij)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            ij = cv2.resize(ij, (32*16, 32*16), interpolation=cv2.INTER_NEAREST)
            img[ij > 0] = img[ij > 0] * 0.6
            img = img + ij * 0.39

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


counts=0
def cvshow_seg_kpt(image, label, masks, keypoints):
    import cv2
    import numpy as np
    keypoints[..., 0] *= 32*16
    keypoints[..., 1] *= 32*16
    keypoints = np.asarray(keypoints, dtype=np.int32)

    global counts
    savepath = os.path.join(abspath, 'datas', 'seg')
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
        kk = label[:, 0].long()==j
        for ij in range(len(masks)):
            if not kk[ij]:
                continue
            ij = masks[ij].cpu().numpy()
            ij = np.array(ij, dtype=np.float32)
            ijb = ij * 200
            ij = np.stack([ijb, ij, ij])
            ij = np.transpose(ij, (1, 2, 0))
            
            # cv2.imshow('img', ij)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            ij = cv2.resize(ij, (32*16, 32*16), interpolation=cv2.INTER_NEAREST)
            img[ij > 0] = img[ij > 0] * 0.6
            img = img + ij * 0.39

        for ij in lis[j]:
            print(ij, img.shape)
            cv2.rectangle(img, (ij[1], ij[2]), (ij[3], ij[4]), [0, 0, 128], 1)
            cv2.putText(img, classes[ij[0]], (ij[1], ij[2]),
                        font, 0.6, (128, 0, 128), 1)
        for ij in range(len(keypoints)):
            if not kk[ij]:
                continue
            kptinst = keypoints[ij]
            kptcolor = [[  0, 255,   0],[  0, 255,   0],[  0, 255,   0],[  0, 255,   0],[  0, 255,   0],[255, 128,   0],[255, 128,   0],
                        [255, 128,   0],[255, 128,   0],[255, 128,   0],[255, 128,   0],[ 51, 153, 255],
                        [ 51, 153, 255],[ 51, 153, 255],[ 51, 153, 255],[ 51, 153, 255],[ 51, 153, 255]]
            for ijw in range(len(kptinst)):
                kpt = kptinst[ijw]
                if kpt[-1]==0:
                    continue
                cv2.circle(img, (kpt[0], kpt[1]), 3, kptcolor[ijw], 1)
            skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
            skeleton_color = [[ 51, 153, 255],[ 51, 153, 255],[ 51, 153, 255],[ 51, 153, 255],[255,  51, 255],[255,  51, 255],[255,  51, 255],
                              [255, 128,   0],[255, 128,   0],[255, 128,   0],[255, 128,   0],[255, 128,   0],[  0, 255,   0],[  0, 255,   0],
                              [  0, 255,   0],[  0, 255,   0],[  0, 255,   0],[  0, 255,   0],[  0, 255,   0]]
            for ind, ske in enumerate(skeleton):
                pot1 = (kptinst[ske[0] - 1][0], kptinst[ske[0] - 1][1])
                pot2 = (kptinst[ske[1] - 1][0], kptinst[ske[1] - 1][1])
                if kptinst[ske[0] - 1][2]==0 or kptinst[ske[1] - 1][2]==0:
                    continue
                cv2.line(img, pot1, pot2, skeleton_color[ind], 1)
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
            if not p.requires_grad:
                continue
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

def collate_fnseg(batch):
    batch = [i for i in batch if i is not None]
    # print(batch)
    img, label, masks, image_id = list(zip(*batch))  # transposed
    # print(len(label))
    for i, l in enumerate(label):
        l[:, 0] = i
    retmask = []
    for i, m in enumerate(masks):
        retmask.extend(m)
    images = torch.stack(img)
    labels = torch.cat(label, 0)
    retmask = torch.stack(retmask)
    return images, labels, retmask, image_id

def collate_fnseg_keypoint(batch):
    batch = [i for i in batch if i is not None]
    # print(batch)
    img, label, masks, keypoint, image_id = list(zip(*batch))  # transposed
    # print(len(label))
    for i, l in enumerate(label):
        l[:, 0] = i
    retmask = []
    for i, m in enumerate(masks):
        retmask.extend(m)
    retkpt = []
    for i, m in enumerate(keypoint):
        retkpt.extend(m)
    images = torch.stack(img)
    labels = torch.cat(label, 0)
    retmask = torch.stack(retmask)
    retkpt = torch.stack(retkpt)
    return images, labels, retmask, retkpt, image_id

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