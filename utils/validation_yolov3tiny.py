#Author：ZouJiu
#Time: 2021-8-13
import os

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import numpy as np
import torch
import cv2
import torch.optim as optim
# from config.config_yolov3tiny import validsave, validpath, classes, batch_size, subsiz, validtruth, inputwidth
from config.config_yolovKKn import validsave, validpath, classes, batch_size, subsiz, validtruth, inputwidth
from utils.utils_yolov3tiny import non_max_suppression, scale_coords, segnon_max_suppression, segnon_keypoint_max_suppression
from mAP.mAP import calculate
from PIL import Image
import tqdm
from utils.evaluate_yolov3tiny import loadeva



def validation_map_segkeypoint(model, yolovfive, dataloader, device, score_thresh_now = 0.001, nms_thresh_now = 0.6, \
                    max_det=300):
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
    # if len(kl)>100:
    #     np.random.shuffle(kl)
    #     kl = kl[:100]
    #     # kl = np.random.choice(kl, 600, replace=False)
    # length = len(kl)
    
    dic, rev_cat = loadeva()
    
    for i, (image, labels, gtmask, keypoint, image_id) in enumerate(tqdm.tqdm(dataloader, mininterval=6)):
        image = image.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction, keypoint, Proto = model(image, yolovfive = yolovfive)   # zip -r -q /home/featurize/work/COCO2017.zip ./COCO2017
        prediction, outkpt = segnon_keypoint_max_suppression(prediction, keypoint, score_thresh_now, nms_thresh_now, max_det=max_det, agnostic=False)

        for j, det in enumerate(prediction):
            id = image_id[j][0]
            name = dic[id]
            ff = open(os.path.join(validsave, name.replace('.jpg', '.txt')), 'w')
            if len(det)==0:
                continue
            det[:, :2*2] = scale_coords(image.shape[2:], det[:, :2*2], image_id[j][1]).round()
            det = det.cpu().numpy()
            for *xyxy, conf in reversed(det[:, :6-1]):
                xmin, ymin, xmax, ymax = xyxy
                ff.write(classes[int(0)]+','+str(xmin)+','+str(ymin)+\
                        ","+str(xmax)+','+str(ymax)+','+str(conf)+'\n')
            ff.close()

    return calculate(validtruth, validsave, classes), len(dataloader) * (batch_size//subsiz)

def validation_map_seg(model, yolovfive, dataloader, device, score_thresh_now = 0.001, nms_thresh_now = 0.6, \
                    max_det=300):
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
    # if len(kl)>100:
    #     np.random.shuffle(kl)
    #     kl = kl[:100]
    #     # kl = np.random.choice(kl, 600, replace=False)
    # length = len(kl)
    
    dic, rev_cat = loadeva()
    
    for i, (image, labels, gtmask, image_id) in enumerate(tqdm.tqdm(dataloader, mininterval=6)):
        image = image.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction, prototype = model(image, yolovfive = yolovfive)   # zip -r -q /home/featurize/work/COCO2017.zip ./COCO2017
        prediction = segnon_max_suppression(prediction, score_thresh_now, nms_thresh_now, max_det=max_det, agnostic=False)

        for j, det in enumerate(prediction):
            id = image_id[j][0]
            name = dic[id]
            ff = open(os.path.join(validsave, name.replace('.jpg', '.txt')), 'w')
            if len(det)==0:
                continue
            det[:, :2*2] = scale_coords(image.shape[2:], det[:, :2*2], image_id[j][1]).round()
            det = det.cpu().numpy()
            for *xyxy, conf, label in reversed(det[:, :6]):
                xmin, ymin, xmax, ymax = xyxy
                ff.write(classes[int(label)]+','+str(xmin)+','+str(ymin)+\
                        ","+str(xmax)+','+str(ymax)+','+str(conf)+'\n')
            ff.close()

    return calculate(validtruth, validsave, classes), len(dataloader) * (batch_size//subsiz)

def validation_map(model, yolovfive, dataloader, device, score_thresh_now = 0.001, nms_thresh_now = 0.6, \
                    max_det=300):
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
    # if len(kl)>100:
    #     np.random.shuffle(kl)
    #     kl = kl[:100]
    #     # kl = np.random.choice(kl, 600, replace=False)
    # length = len(kl)
    
    dic, rev_cat = loadeva()
    
    for i, (image, labels, image_id) in enumerate(tqdm.tqdm(dataloader, mininterval=6)):
        image = image.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            prediction = model(image, yolovfive = yolovfive)   # zip -r -q /home/featurize/work/COCO2017.zip ./COCO2017
        prediction = non_max_suppression(prediction, score_thresh_now, nms_thresh_now, max_det=max_det, agnostic=False)

        for j, det in enumerate(prediction):
            id = image_id[j][0]
            name = dic[id]
            ff = open(os.path.join(validsave, name.replace('.jpg', '.txt')), 'w')
            if len(det)==0:
                continue
            det[:, :2*2] = scale_coords(image.shape[2:], det[:, :2*2], image_id[j][1]).round()
            det = det.cpu().numpy()
            for *xyxy, conf, label in reversed(det):
                xmin, ymin, xmax, ymax = xyxy
                ff.write(classes[int(label)]+','+str(xmin)+','+str(ymin)+\
                        ","+str(xmax)+','+str(ymax)+','+str(conf)+'\n')
            ff.close()

    return calculate(validtruth, validsave, classes), len(dataloader) * (batch_size//subsiz)

    # for i, pth in enumerate(kl):
    #     nam = pth.split(os.sep)[-1]
    #     image = Image.open(pth).convert("RGB")
    #     w, h = image.size
    #     cvimg = np.asarray(image)
    #     img = TFRESIZE(image)
    #     img = torch.unsqueeze(img, 0).to(device)
    #     with torch.no_grad():
    #         # print(model.training)
    #         model = model.to(device)
    #         img = img.to(device)
    #         pred = model(img)
    #         pred = non_max_suppression(pred, score_thresh, nms_thresh, agnostic=False)
    #         tails = nam[-6:].split('.')[-1]
    #         ff = open(os.path.join(validsave, nam.replace(tails, 'txt')), 'w')
    #         for i, det in enumerate(pred):
    #             # Rescale boxes from img_size to im0 size
    #             if len(det)==0:
    #                 continue
    #             det[:, :4] = scale_coords(img.shape[2:], det[:, :4], cvimg.shape).round()
    #             for *xyxy, conf, label in reversed(det):
    #                 xmin, ymin, xmax, ymax = xyxy
    #                 xmin, ymin, xmax, ymax = xmin.cpu().item(), ymin.cpu().item(), xmax.cpu().item(), ymax.cpu().item()
    #                 conf = conf.cpu().item()
    #                 label = label.cpu().item()
    #                 # cxp, cyp, wp, hp, maxscore, label = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]
    #                 # xmin = (cxp - wp/2)*w
    #                 # ymin = (cyp - hp/2)*h
    #                 # xmax = (cxp + wp/2)*w
    #                 # ymax = (cyp + hp/2)*h
    #                 # cvfont = cv2.FONT_HERSHEY_SIMPLEX
    #                 # image = np.asarray(image)
    #                 # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #                 # print(cxp, cyp, wp, hp, maxscore, label)
    #                 try:
    #                     minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin))), min(h-1, max(0, int(ymin))), \
    #                         min(w-1, max(0, int(xmax))), min(h-1, max(0, int(ymax)))
    #                 except:
    #                     pass
    #                 ff.write(classes[int(label)]+','+str(minx)+','+str(miny)+\
    #                     ","+str(maxx)+','+str(maxy)+','+str(conf)+'\n')
    #                 # try:
    #                 #     cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 1)
    #                 # except Exception as e:
    #                 #     print(e)
    #                 #     continue
    #                 # text = classes[int(label[j])] + ' ' + str(round(maxscore[j].item(),3))
    #                 # cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 255], 1)
    #             ff.close()
    # return calculate(validtruth, validsave, classes), length


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