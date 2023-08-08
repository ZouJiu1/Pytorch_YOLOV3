#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import cv2
import sys
nowpath = os.path.abspath("./")
sys.path.append(nowpath)
import torch.optim as optim
from config.config_yolov3_20230727 import *
from utils.utils_yolov3tiny import non_max_suppression, scale_coords
from mAP.mAP import calculate
from PIL import Image

def validation_map(model):
    model.eval()
    if not os.path.exists(validsave):
        os.mkdir(validsave)
    for i in os.listdir(validsave):
        os.remove(os.path.join(validsave, i))
    kl = []
    with open(validpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            if os.path.exists(i):
                kl.append(i)
    if len(kl)>100:
        np.random.shuffle(kl)
        kl = kl[:100]
        # kl = np.random.choice(kl, 600, replace=False)
    length = len(kl)
    for i, pth in enumerate(kl):
        nam = pth.split(os.sep)[-1]
        image = Image.open(pth).convert("RGB")
        w, h = image.size
        cvimg = np.asarray(image)
        img = TFRESIZE(image)
        img = torch.unsqueeze(img, 0).to(device)
        with torch.no_grad():
            # print(model.training)
            model = model.to(device)
            img = img.to(device)
            pred = model(img)
            pred = non_max_suppression(pred, score_thresh, nms_thresh, agnostic=False)
            tails = nam[-6:].split('.')[-1]
            ff = open(os.path.join(validsave, nam.replace(tails, 'txt')), 'w')
            for i, det in enumerate(pred):
                # Rescale boxes from img_size to im0 size
                if len(det)==0:
                    continue
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cvimg.shape).round()
                for *xyxy, conf, label in reversed(det):
                    xmin, ymin, xmax, ymax = xyxy
                    xmin, ymin, xmax, ymax = xmin.cpu().item(), ymin.cpu().item(), xmax.cpu().item(), ymax.cpu().item()
                    conf = conf.cpu().item()
                    label = label.cpu().item()
                    # cxp, cyp, wp, hp, maxscore, label = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]
                    # xmin = (cxp - wp/2)*w
                    # ymin = (cyp - hp/2)*h
                    # xmax = (cxp + wp/2)*w
                    # ymax = (cyp + hp/2)*h
                    # cvfont = cv2.FONT_HERSHEY_SIMPLEX
                    # image = np.asarray(image)
                    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # print(cxp, cyp, wp, hp, maxscore, label)
                    try:
                        minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin))), min(h-1, max(0, int(ymin))), \
                            min(w-1, max(0, int(xmax))), min(h-1, max(0, int(ymax)))
                    except:
                        pass
                    ff.write(classes[int(label)]+','+str(minx)+','+str(miny)+\
                        ","+str(maxx)+','+str(maxy)+','+str(conf)+'\n')
                    # try:
                    #     cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 1)
                    # except Exception as e:
                    #     print(e)
                    #     continue
                    # text = classes[int(label[j])] + ' ' + str(round(maxscore[j].item(),3))
                    # cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 255], 1)
                ff.close()
    return calculate(validtruth, validsave, classes), length


def validation_map_myself(model, validloader):
    model.eval()
    if not os.path.exists(validsave):
        os.mkdir(validsave)
    for i in os.listdir(validsave):
        os.remove(os.path.join(validsave, i))
    kl = []
    with open(validpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            if os.path.exists(i):
                kl.append(i)
    if len(kl)>600:
        kl = np.random.choice(kl, 600, replace=False)
    length = len(kl)
    for i, pth in enumerate(kl):
        nam = pth.split(os.sep)[-1]
        image = Image.open(pth).convert("RGB")
        w, h = image.size
        image = TF(image)
        image = torch.unsqueeze(image, 0).to(device)

        p = model(image, [])[0]
        if p.shape[1]==0:
            continue
        p = p[0]
        cxp, cyp, wp, hp, maxscore, label = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]
        xmin = (cxp - wp/2)*w
        ymin = (cyp - hp/2)*h
        xmax = (cxp + wp/2)*w
        ymax = (cyp + hp/2)*h
        # cvfont = cv2.FONT_HERSHEY_SIMPLEX
        # image = np.asarray(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        tails = nam[-6:].split('.')[-1]
        ff = open(os.path.join(validsave, nam.replace(tails, 'txt')), 'w')
        for j in range(len(label)):
            # print(cxp, cyp, wp, hp, maxscore, label)
            try:
                minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin[j]))), min(h-1, max(0, int(ymin[j]))), \
                    min(w-1, max(0, int(xmax[j]))), min(h-1, max(0, int(ymax[j])))
            except:
                continue
            ff.write(classes[int(label[j])]+','+str(minx)+','+str(miny)+\
                ","+str(maxx)+','+str(maxy)+','+str(maxscore[0].item())+'\n')
            # try:
            #     cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 1)
            # except Exception as e:
            #     print(e)
            #     continue
            # text = classes[int(label[j])] + ' ' + str(round(maxscore[j].item(),3))
            # cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 255], 1)
        ff.close()
    return calculate(validtruth, validsave, classes)