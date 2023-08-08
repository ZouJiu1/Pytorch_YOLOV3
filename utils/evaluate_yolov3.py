import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

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
from models.Yolov3_20230727 import Yolov3Net
pylab.rcParams['figure.figsize'] = (10.0, 8.0)
from config.config_yolov3tiny import *
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

def evaluation(dataloader = None, model=None, score_thresh_now = 0.001, nms_thresh_now = 0.6, \
            length = 0, max_det=300, cvshow = False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not model:
        model = Yolov3Net(num_classes, anchors, device, inputwidth)
        
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval()
    # score_thresh = 0.6
    # nms_thresh = 0.6 - 0.2 - 0.1
    dic, rev_cat = loadeva()

    if not dataloader:
        valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                augment = False, inputwidth = inputwidth, transform=TFRESIZE)
        dataloader = DataLoader(valdata, batch_size = batch_size // subsiz, shuffle=True, \
            num_workers=cpu_count(), collate_fn=collate_fn)

    result = []
    annFile = os.path.join(abspath, 'datas', 'instances_val2017.json')
    cocoGt=COCO(annFile)
    
    with torch.no_grad():
        model(torch.rand(1, 3, 32*16, 32*16).to(device))

    pthshow = os.path.join(abspath, 'datas', 'imshow')
    for i in os.listdir(pthshow):
        os.remove(os.path.join(pthshow, i))

    for i, (image, labels, image_id) in enumerate(tqdm.tqdm(dataloader)):
        image = image.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction = model(image)   # zip -r -q /home/featurize/work/COCO2017.zip ./COCO2017
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
                cla = classes[int(label)]
                cvshow_label.append([cla, int(xmin), int(ymin), int(xmax), int(ymax)])
                cla_id = rev_cat[cla]
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

if __name__ == "__main__":
    map, mapfivezero = evaluation()
    k = 0