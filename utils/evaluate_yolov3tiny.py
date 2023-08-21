import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import cv2
import json
import torch
import numpy as np

from loaddata.cocoread import trainDataset
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
# from models.Yolov3tiny import yolov3tinyNet
from models.Yolovkkn import YolovKKNet
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
# from config.config_yolov3tiny import *
from config.config_yolovKKn import *
from multiprocessing import cpu_count
from utils.common import cvshow_, collate_fn
from torch.utils.data import Dataset, DataLoader
from utils.utils_yolov3tiny import non_max_suppression, loadtorchmodel, scale_coords
import tqdm

def loadeva():
    # cat = {}
    rev_cat = {}
    with open(os.path.join(abspath, 'datas', 'cococat.json'), 'r') as obj:
        f = json.load(obj)
        for key, value in f.items():
            id = value['id']
            name = value['name']
            # cat[id] = name
            rev_cat[name] = id

    with open(pth_evaluate, 'r') as f:
        jf = json.load(f)

    dic = {}
    for i in range(len(jf['images'])):
        nam = jf['images'][i]['file_name']
        id  = jf['images'][i]['id']
        dic[id] = nam

    return dic, rev_cat

def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

def evaluation(pretrainedmodel, dataloader = None, model=None, score_thresh_now = 0.001, nms_thresh_now = 0.6, \
            length = 0, max_det=300, cvshow = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    img_evaluate = r'/root/project/val2017'
    cocolabel = coco80_to_coco91_class()
    if not model:
        # model = yolov3tinyNet(num_classes, anchors, device, inputwidth)
        model = YolovKKNet(num_classes, anchors, device, inputwidth)
        
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cpu'))
        kkk = {}
        for key, value in state_dict['state_dict'].items():
            kkk[key.replace("module.", "")] = value
        state_dict['state_dict'] = kkk
        pretrained = state_dict['state_dict']
        model.load_state_dict(pretrained, strict=True)
        del state_dict, pretrained, kkk

    model.to(device)
    model.eval()
    # score_thresh = 0.6
    # nms_thresh = 0.6 - 0.2 - 0.1
    dic, rev_cat = loadeva()
    pth_evaluate = r'/root/project/yolov5-master/coco/annotations/instances_val2017.json'
    img_evaluate = r'/root/project/yolov5-master/coco/images/val2017'
    if not dataloader:
        valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                augment = False, inputwidth = inputwidth, transform=TFRESIZE, evaluate = True)
        dataloader = DataLoader(valdata, batch_size = 10, shuffle=False, \
            num_workers=11, collate_fn=collate_fn)

    result = []
    annFile = os.path.join(abspath, 'datas', 'instances_val2017.json')
    cocoGt=COCO(annFile)
    
    with torch.no_grad():
        model(torch.rand(1, 3, 32*16, 32*16).to(device))

    pthshow = os.path.join(abspath, 'datas', 'imshow')
    for i in os.listdir(pthshow):
        os.remove(os.path.join(pthshow, i))
    if chooseLoss in ["20230730", "yolofive"]:
        yolovfive = True
    for i, (image, _, image_id) in enumerate(tqdm.tqdm(dataloader)):
        image = image.to(device)
        with torch.no_grad():
            prediction = model(image, yolovfive = yolovfive)   # zip -r -q /home/featurize/work/COCO2017.zip ./COCO2017
        prediction = non_max_suppression(prediction, score_thresh_now, nms_thresh_now, max_det=max_det, agnostic=False)

        for j, det in enumerate(prediction):
            id = image_id[j][0]
            name = dic[id]
            if len(det)==0:
                continue
            det[:, :2*2] = scale_coords(image.shape[2:], det[:, :2*2], image_id[j][1]).round()
            det = det.cpu().numpy()
            cvshow_label = []
            for *xyxy, conf, label in reversed(det):
                det_single = {"image_id": id}
                xmin, ymin, xmax, ymax = xyxy
                w  = xmax - xmin
                h  = ymax - ymin
                cla_id = cocolabel[int(label)]
                # cla = classes[int(label)]
                # cvshow_label.append([cla, int(xmin), int(ymin), int(xmax), int(ymax)])
                # cla_id = rev_cat[cla]
                det_single['category_id'] = cla_id
                det_single['bbox'] = [np.float64(round(xmin, 2)), np.float64(round(ymin, 2)), np.float64(round(w, 2)), np.float64(round(h, 2))]
                det_single['score'] = np.float64(round(conf, 3))
                result.append(det_single)

            if cvshow:
                font = cv2.FONT_HERSHEY_SIMPLEX
                # img = image[j].detach().cpu().numpy()
                # img = np.transpose(img, (1, 2, 0))
                # img = np.array(img*255, dtype=np.uint8)
                # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = cv2.imread(os.path.join(img_evaluate, name))
                for ij in cvshow_label:
                    print(ij, img.shape)
                    cv2.rectangle(img, (ij[1], ij[2]), (ij[3], ij[4]), [0, 0, 128], 1)
                    cv2.putText(img, ij[0], (ij[1], ij[2]),
                                font, 0.6, (128, 0, 128), 1)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                cv2.imwrite(os.path.join(abspath, 'datas', 'imshow', str(j) + '.jpg'), img)
        if cvshow:
            exit(0)

    with open(os.path.join(abspath, 'datas', 'instances_result_val2017.json'), 'w') as obj:
        json.dump(result, obj)
    annType = 'bbox'
    cocoDt=cocoGt.loadRes(os.path.join(abspath, 'datas', 'instances_result_val2017.json'))
    cocoEval = COCOeval(cocoGt,cocoDt,annType)

    imgIds=sorted(cocoGt.getImgIds())
    if length!=0:
        imgIds=imgIds[0:100]

    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    map, mapfivezero = cocoEval.stats[:2]
    print(map, mapfivezero)
    return map, mapfivezero


def apcalcul():
    pth = os.path.join(abspath, 'datas', 'cocoval', 'truth')
    valpth = os.path.join(abspath, 'datas', 'instances_val2017.json')

    with open(valpth, 'r') as obj:
        jf = json.load(obj)

    dic = {}
    for i in range(len(jf['images'])):
        nam = jf['images'][i]['file_name']
        # if nam == '000000090026.jpg':
        #     k = 0
        id  = jf['images'][i]['id']
        width = jf['images'][i]['width']
        height = jf['images'][i]['height']
        dic[id] = [nam, height, width]

    cat = {}
    with open(os.path.join(abspath, 'datas', 'cococat.json'), 'r') as obj:
        f = json.load(obj)
        for key, value in f.items():
            id = value['id']
            name = value['name']
            cat[id] = name

    for i in range(len(jf['annotations'])):
        id = jf['annotations'][i]['image_id']
        label_id = jf['annotations'][i]['category_id']
        # label = classes.index(cat[label_id])
        label = cat[label_id]
        xmin, ymin, w, h = jf['annotations'][i]['bbox']
        xmax = xmin + w
        ymax = ymin + h

        dic[id].append([label, xmin, ymin, xmax, ymax])

    # delete = []

    # for key, value in dic.items():
    #     if len(value) == 3:
    #         delete.append(key)

    # for i in delete:
    #     dic.pop(i)
    
    for key, value in dic.items():
        label = value[3:]
        txtpth = os.path.join(pth, value[0].replace(".jpg", ".txt"))
        if len(label) == 0:
            f = open(txtpth, 'w')
            f.close()
            continue
        with open(txtpth, 'w') as obj:
            for i in range(len(label)):
                l, xmin, ymin, xmax, ymax = label[i]
                xmin, ymin, xmax, ymax = str(round(xmin, 2)), str(round(ymin, 2)), str(round(xmax, 2)), str(round(ymax, 2))
                kk = ','.join([l, xmin, ymin, xmax, ymax])
                obj.write(kk + "\n")

def genval():
    pth = r'/home/featurize/data/COCO2017/val2017'
    with open(r'/home/featurize/work/Pytorch_YOLOV3/datas/cocoval.txt', 'w') as obj:
        for i in os.listdir(pth):
            obj.write(os.path.join(pth, i) + "\n")

if __name__ == "__main__":
    pth = r'/root/project/yolovkkn/2023-08-14yolokkn'
    for r, d, f in os.walk(pth):
        for i in f:
            record = os.path.join(r, 'record')
            if not os.path.exists(record) or os.path.getsize(record)==0:
                os.system("touch %s" % record)
            else:
                break
            if '.pth' in i:
                os.system(" ls -l %s " % os.path.join(r, i))
                map, mapfivezero = evaluation(os.path.join(r, i))
    # apcalcul()
    # genval()
    k = 0