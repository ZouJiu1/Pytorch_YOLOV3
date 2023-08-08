#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import cv2
import sys
sys.path.append(r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3')
from config.config import *
from mAP.mAP import calculate

def validation_map(model, validloader):
    model.eval()
    for i in os.listdir(validsave):
        os.remove(os.path.join(validsave, i))
    for i, (image, nam) in enumerate(validloader):
        if i > 300:
            break
        b, c, h, w = image.size()
        p = model(image)[0]
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
        tails = nam[0][-6:].split('.')[-1]
        ff = open(os.path.join(validsave, nam[0].replace(tails, 'txt')), 'w')
        for j in range(len(label)):
            print(label)
            minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin[j]))), min(h-1, max(0, int(ymin[j]))), \
                min(w-1, max(0, int(xmax[j]))), min(h-1, max(0, int(ymax[j])))
            ff.write(classes[int(label[j])]+' '+str(minx)+' '+str(miny)+\
                " "+str(maxx)+' '+str(maxy)+' '+str(maxscore))
            # try:
            #     cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 1)
            # except Exception as e:
            #     print(e)
            #     continue
            # text = classes[int(label[j])] + ' ' + str(round(maxscore[j].item(),3))
            # cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 255], 1)
        ff.close()
    return calculate(validtruth, validsave, classes)

