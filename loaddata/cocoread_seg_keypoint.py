#Author：ZouJiu
#Time: 2022-12-10
import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)
from config.config_yolov3_20230727 import train_imgpath
import torch
import cv2
import numpy as np
import json
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
# Declare an augmentation pipeline
P = 0.2
AAAtransform = A.Compose([
    A.RandomGamma(p=P),
    A.HueSaturationValue(p=P),
    A.RandomBrightnessContrast(p=P),
    A.MotionBlur(p=P),
    A.GaussianBlur(p=P),
    A.GaussNoise(p=P),
    # A.ToGray(p=P),
    A.Equalize(p=P),
    A.PixelDropout(p=P),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def polygon2mask(img_size, inputwidth, polygons, cx, cy, id, xmin, ymin, xmax, ymax, color=1, downsample_ratio=1):
    #  https://github.com/ultralytics/yolov5/tree/master/utils
    """
    Args:
        img_size (tuple): The image size.
        polygons (np.ndarray): [N, M], N is the number of polygons,
            M is the number of points(Be divided by 2).
    """
    # downsample_ratio = 2*2
    mask = np.zeros(img_size, dtype=np.uint8)
    for i in range(len(polygons)):
        polyg = np.asarray(polygons[i])
        polyg = polyg.astype(np.int32)
        shape = polyg.shape
        # polyg = polyg.reshape(-1, 2)
        # cv2.fillConvexPoly(mask, polyg, color=color)

        polyg = polyg.reshape(1, -1, 2)
        cv2.fillPoly(mask, polyg, color=color)
    nh, nw = (inputwidth // downsample_ratio, inputwidth // downsample_ratio)
    # NOTE: fillPoly firstly then resize is trying the keep the same way
    # of loss calculation when mask-ratio=1.
    mask = cv2.resize(mask, (nw, nh))
    mask = np.array(mask, dtype=np.bool_)
    return mask

class trainDataset(Dataset):
    def __init__(self, traintxt, all_imgpath, stride, anchors, inputwidth, augment=True, transform=None, target_transform=None, evaluate = False):
        super(trainDataset, self).__init__()
        self.traintxt = traintxt
        self.transform = transform
        self.target_transform = target_transform
        self.trainpath = []
        self.stride = stride
        self.anchors = anchors
        self.inputwidth = inputwidth
        self.augment = augment
        self.all_imgpath = all_imgpath

        with open(self.traintxt, 'r') as f:
            jf = json.load(f)

        # with open(r'C:\Users\10696\Desktop\Pytorch_YOLOV3\cocoapi_master\cocoval.json', 'w') as f:
        #     json.dump(jf, f, indent=2, separators=(":", ";"))

        # cat = {}
        # with open(r'C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\coconame2017.txt', 'w') as objkkk:
        #     with open(r'C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\cococat.json', 'r') as obj:
        #         f = json.load(obj)
        #         for key, value in f.items():
        #             id = value['id']
        #             name = value['name']
        #             cat[id] = name
        #     # for i in range(1, 91):
        #     #     if i not in cat.keys():
        #     #         cat[i] = "_____placeholder______"
        #     cat = sorted(cat.items(), key = lambda k : k[0])
        #     for i in cat:
        #         objkkk.write(i[1] +"\n")

        cat = {}
        classes = []
        with open(os.path.join(abspath, 'datas', 'cococat.json'), 'r') as obj:
            f = json.load(obj)
            for key, value in f.items():
                id = value['id']
                name = value['name']
                cat[id] = name

        with open(os.path.join(abspath, 'datas', 'coconame2017.txt'), 'r', encoding='utf-8') as obj:
            for i in obj.readlines():
                i = i.strip()
                classes.append(i)

        dic = {}
        for i in range(len(jf['images'])):
            nam = jf['images'][i]['file_name']
            # if nam == '000000090026.jpg':
            #     k = 0
            id  = jf['images'][i]['id']
            width = jf['images'][i]['width']
            height = jf['images'][i]['height']
            dic[id] = [nam, height, width]

        self.count = {}
        allnum = 0
        for i in range(len(jf['annotations'])):
            id = jf['annotations'][i]['image_id']
            if jf['annotations'][i]['num_keypoints']==0:
                continue
            keypoints = np.array(jf['annotations'][i]['keypoints'], dtype=np.float32)
            keypoints = keypoints.reshape((16+1, 3))
            
            label_id = jf['annotations'][i]['category_id']
            label = classes.index(cat[label_id])
            if label!=0:
                continue

            if label not in self.count.keys():
                self.count[label] = [1]
            else:
                self.count[label][0] += 1
            allnum += 1

            polygon = jf['annotations'][i]['segmentation']
            if not isinstance(polygon, list):
                continue
            imgsize = [dic[id][1], dic[id][2]]
            # imgsize = [dic[id][2], dic[id][1]]
            # box = jf['annotations'][i]['bbox']
            segment = []
            for ip in range(len(polygon)):
                segment += polygon[ip]
            segment = np.array(segment, dtype=np.float32).reshape((-1, 2))
            x, y = segment.T
            xmin, ymin, xmax, ymax = x.min(), y.min(), x.max(), y.max()
            
            cx = (xmin + xmax) / 2 / dic[id][2]
            cy = (ymin + ymax) / 2 / dic[id][1]
            w = (xmax - xmin) / dic[id][2]
            h = (ymax - ymin) / dic[id][1]
            
            masks = polygon2mask(imgsize, self.inputwidth, polygon, (xmin + xmax) / 2, (ymin + ymax) / 2, id, xmin, ymin, xmax, ymax, color=1, downsample_ratio=2*2)
            keypoints[:, 0] = keypoints[:, 0] / dic[id][2]
            keypoints[:, 1] = keypoints[:, 1] / dic[id][1]
            if cx > 1.0 or cx < 0.0 or cy > 1.0 or cy < 0.0 or w > 1.0 or w <= 0.0 or h > 1.0 or h <= 0.0:
                continue

            dic[id].append([label, cx, cy, w, h, masks, keypoints])
        
        for key, value in self.count.items():
            self.count[key].extend([self.count[0][0] / value[0], allnum])

        count_scale_pth = os.path.join(abspath, 'datas', 'cococount.json')
        if os.path.exists(count_scale_pth):
            with open(count_scale_pth, 'r', encoding='utf-8') as obj:
                self.co = json.load(obj)
            self.count = {}
            for key, value in self.co.items():
                self.count[int(key)] = list(value)

        self.count_scale = torch.zeros((len(classes)))
        for i in range(len(classes)):
            self.count_scale[i] = float(self.count[i][1])

        # with open(count_scale_pth, 'w', encoding='utf-8') as obj:
        #     json.dump(self.count, obj, indent = 2, separators=(",", ":"))
        # k = dic[90026]

        delete = []
        for key, value in dic.items():
            if len(value)==3:
                if evaluate:
                    dic[key] = value + [[0, 0.6, 0.6, 0.01, 0.06]]
                else:
                    delete.append(key)
        for i in delete:
            dic.pop(i)
        self.trainpath = sorted(dic.items(), key = lambda k : k[0])
        
        # pth = r'/root/autodl-tmp/labels'
        # for idx in range(len(self.trainpath)):
        #     choose = self.trainpath[idx][1]
        #     alllabels = choose[3:]
        #     txtpath = os.path.join(pth, choose[0].replace(".jpg", ".txt"))
        #     with open(txtpath, 'w') as obj:
        #         for i in range(len(alllabels)):
        #             label, cx, cy, w, h = alllabels[i]
        #             obj.write(str(int(label)) + ' ' + str(cx) + ' ' + str(cy) + " " + str(w) + " " + str(h) + "\n")
        del dic, jf

    def __len__(self):
        return len(self.trainpath)
    
    def __getitem__(self, idx):
        # inpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\checkpoints'
        choose = self.trainpath[idx][1]
        imageid = self.trainpath[idx][0]
        height = choose[1]
        width = choose[2]
        imgpath = os.path.join(self.all_imgpath, choose[0])
        # imgpath = r'F:\\20230416\\coco\\images\\val2014\\COCO_val2014_000000496575.jpg'
        image = cv2.imread(imgpath)
        imageid = [imageid, (height, width)]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # labelpath = imgpath.replace('JPEGImages','labels').replace('images', 'labels').replace(".jpg", '.txt')
        bboxes = []
        labels = []
        gt = []
        masks = []
        kpt = []
        # print(labelpath, imgpath)
        
        # outpath = r'/root/project/Pytorch_YOLOV3/loaddata/saveimg'
        # cvfont = cv2.FONT_HERSHEY_SIMPLEX
        
        alllabels = choose[3:]
        for i in range(len(alllabels)):
            label, cx, cy, w, h, mask, keypoints = alllabels[i]
            if len(alllabels) > 10 and w > 0.7 and h < 0.7 and label==0:
                # xmin = int((cx - w/2) * 32*16)
                # ymin = int((cy - h/2) * 32*16)
                # xmax = int((cx + w/2) * 32*16)
                # ymax = int((cy + h/2) * 32*16)
                # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [255, 0, 0], 2)
                # cv2.putText(image, str(label), (xmin, ymin+13), cvfont, 1, [255, 0, 0], 1)
                # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(os.path.join(outpath, str(imageid[0])+".jpg"), img)
                continue
            # label, cx, cy, w, h = int(label), float(cx), float(cy), float(w), float(h)
            # mask[mask > 0] = 200
            # cv2.imshow('name', mask)
            # cv2.waitKey(0)
            masks.append(mask)
            kpt.append(keypoints)
            gt.append([0, int(label), cx, cy, w, h])
            bboxes.append([cx, cy, w, h])
            labels.append(int(label))
        bboxes = np.array(bboxes)
        # print(bboxes)
        if self.target_transform:
            bboxes = self.target_transform(bboxes)
        if self.augment:
            try:
                transformed = AAAtransform(image=image, bboxes = bboxes, class_labels = labels, )
                image = transformed['image']
                gtt = transformed['bboxes']
                class_labels = transformed['class_labels']
                kkk = []
                for ind, i in enumerate(gtt):
                    kkk.append([0, class_labels[ind], i[0], i[1], i[2], i[3]])
                gt = np.array(kkk)
            except Exception as e:
                print("e")
                gt = np.array(gt)
                pass
        else:
            gt = np.array(gt)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        masks = np.array(masks)
        kpt = np.array(kpt)
        return image, torch.tensor(gt), torch.tensor(masks, dtype=torch.bool), torch.tensor(kpt, dtype=torch.float32), imageid

if __name__ == '__main__':
    # trainpath = r'F:\annotations_trainval2017\annotations\instances_train2017.json'
    trainpath = r'F:\annotations_trainval2017\annotations\person_keypoints_val2017.json'
    imgpth = r'F:\val201seven'
    inputwidth = 32 * 16
    anchors = [[[10,13], [16,30], [33,23]],\
        [[30,61],  [62,45],  [59,119]],  \
        [[116,90],  [156,198],  [373,326]]]
    strides = [8, 16, 32]
    # anchors = np.array(anchors, dtype = np.float32)
    # print(anchors.shape)
    # for i in range(3):
    #     anchors[i, ...] = anchors[i, ...]/strides[i]
    # print(anchors)
    # exit(0)
    anchor_per_layer = 3
    num_classes = 2       # voc2007_2012 20
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from torch.utils.data import Dataset, DataLoader
    from utils.common import cvshow_, collate_fn
    from config.config_yolov3tiny import TF
    from tqdm import tqdm
    traindata = trainDataset(trainpath, imgpth, stride = strides, anchors = anchors, \
                             inputwidth = inputwidth, transform=TF)
    dataloader = DataLoader(traindata, batch_size = 100,shuffle=False, \
        num_workers=30, collate_fn=collate_fn)
    for i, (images, labels, imageid) in enumerate(tqdm(dataloader)):
        k = 0
        # print(images.size(), labels.size())
