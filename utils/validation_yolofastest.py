#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import cv2
import sys
nowpath = os.path.abspath("./")
sys.path.append(nowpath)
from config.config730_yolofastest import *
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
    if len(kl)>600:
        kl = np.random.choice(kl, 600, replace=False)
    length = len(kl)
    for i, pth in enumerate(kl):
        nam = pth.split(os.sep)[-1]
        image = Image.open(pth).convert("RGB")
        w, h = image.size
        image = TF(image)
        image = torch.unsqueeze(image, 0).to(device)

        p = model(image, True)[0]
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
                pass
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


