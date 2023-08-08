import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config.config717_730 import inputwidth, classes

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


def collate_fn(batch):
    img, label = zip(*batch)  # transposed
    for i, l in enumerate(label):
        try:
            #print(111, l.size())
            l[:, -1] = i  # add target image index for build_targets()
        except Exception as e:
            l = torch.Tensor([[0,0.5,0.1,0.1,0.1,i]])
            print(999, l.size())
    #print(333, torch.cat(label, 0).size(), e) 
    return torch.stack(img, 0), torch.cat(label, 0)

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