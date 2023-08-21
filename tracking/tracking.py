import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
import imageio
sys.path.append(abspath)

from trackers.track import *

import numpy as np
import torch
import time
import cv2
import torchvision

from models.Yolovkkn import YolovKKNet
from PIL import Image
from models.layer_yolo import xywh2xyxy
from config.config_yolovKKn import *
from copy import deepcopy
from utils.utils_yolov3tiny import non_max_suppression, loadtorchmodel, scale_coords

def loading_model():
    model = YolovKKNet(num_classes, anchors, device, inputwidth)
    # pretrainedmodel = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\log\yolovkkn\2023-08-16yolokkn\model_e51_map[0.48195__0.256185]_l64988.657_2023-08-16.pth'
    pretrainedmodel = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\log\yolovkkn\2023-08-18yolokkn\model_e30_map[0.396382__0.245314]_l8.850_2023-08-18.pt'
    if torch.cuda.is_available():
        state_dict = torch.load(pretrainedmodel, map_location = torch.device('cuda'))
    else:
        state_dict = torch.load(pretrainedmodel, map_location = torch.device('cpu'))

    if pretrainedmodel.endswith(".pt"):
        model = state_dict['ema'] if ('ema' in state_dict.keys() and state_dict['ema']!='') else state_dict['state_dict']
        model = model.float().to(device)
        model.device = device
    else:
        kkk = {}
        param = state_dict['ema'] if ('ema' in state_dict.keys() and state_dict['ema']!='') else state_dict['state_dict']
        for key, value in param.items():
            kkk[key.replace("module.", "")] = value
        model.load_state_dict(kkk, strict=True)
        del kkk
    del state_dict
    # pretrained = loadtorchmodel(pretrainedmodel)
    # model.load_state_dict(pretrained, strict=True)
    for i in range(len(model.yolo)):
        model.yolo[i].eval()
        model.yolo[i].device = device
    return model.eval()

def get_source(inpath):
    kk = 0
    if os.path.isdir(inpath):
        kk = os.listdir(inpath)
        kk.sort()
        for i in kk:
            image = cv2.imread(os.path.join(inpath, i))
            yield image
    elif os.path.isfile(inpath):
        tails = inpath.split(".")[-1]
        kk = []
        if tails=='txt':
            with open(inpath, 'r') as obj:
                for i in obj.readlines():
                    kk.append(os.path.join(inpath, i))
            for i in kk:
                image = cv2.imread(i)
                yield image
        else:
            vid = cv2.VideoCapture(inpath)
            marked, frame = vid.read()
            # count = 1
            while marked:
                vid.grab()
                marked, frame = vid.retrieve()
                yield frame
            vid.release()

def trackingsomething():
    model = loading_model()
    yolovfive = True if chooseLoss in ["20230730", "yolofive"] else False
    inpath = r"F:\MOT\MOT16-06-raw.webm"
    # inpath = r"F:\MOT\DETRAC-test-data\Insight-MVT_Annotation_Test\MVI_39031"
    num = 0
    cnt = 0
    score_thresh = 0.1
    cvfont = cv2.FONT_HERSHEY_SIMPLEX
    iterk = get_source(inpath)

    bs = 1
    predictor = on_predict_start(bs)
    framesall = []
    for image in iterk:
        image_origin = deepcopy(image)
        if num==703:
            kk = 0
        if num==300:
            break
        num += 1
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            continue
        h, w, c = image.shape
        image = cv2.resize(image, (inputwidth, inputwidth)) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            pred  = model(image, yolovfive = yolovfive)
        pred = non_max_suppression(pred, score_thresh, nms_thresh, agnostic=False)
        for j, det in enumerate(pred):
            det[:, :4] = scale_coords([inputwidth, inputwidth], det[:, :4], image_origin.shape).round()
            det = det.cpu().numpy()
            predictor.batch_add(image_origin)
            predictor.results[j].orishape = image_origin.shape
            boxes = ALL_box(det, image_origin.shape)
            boxes.cls = det[:, -1]
            boxes.xyxy = det[:, :2*2]
            boxes.conf = det[:, 2*2]
            predictor.results[j].boxes = boxes
        on_predict_postprocess_end(predictor)
        for j, _ in enumerate(range(bs)):
            det = predictor.results[j].boxes.boxes
            if len(det)==0:
                continue
            for *xyxy, id, conf, label in reversed(det):
                try:
                    xmin, ymin, xmax, ymax = xyxy
                except:
                    continue
                xmin, ymin, xmax, ymax = xmin.cpu().item(), ymin.cpu().item(), xmax.cpu().item(), ymax.cpu().item()
                conf = conf.cpu().item()
                label = label.cpu().item()
                minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin))), min(h-1, max(0, int(ymin))), \
                    min(w-1, max(0, int(xmax))), min(h-1, max(0, int(ymax)))
                areak = (maxx - minx) * (maxy - miny)
                kk = np.prod(image_origin.shape[:-1])
                k = areak / kk
                if k > (1-0.1) and len(det) >= 2:
                    continue
                try:
                    cv2.rectangle(image_origin, (minx, miny), (maxx, maxy), [255, 0, 0], 1, lineType=cv2.LINE_AA)
                    cnt += 1
                except Exception as e:
                    print(e)
                    continue
                text = str(int(id.item())) + " " + classes[int(label)] # + ' ' + str(round(conf,1))
                # text = text.replace('4','9')
                cv2.putText(image_origin, text, (minx, miny+13), cvfont, 1, [0, 0, 255], 2, lineType=cv2.LINE_AA)

        # cv2.imshow(inpath, image_origin)
        # cv2.waitKey(1)
        imgtmp = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
        h, w, c = imgtmp.shape
        # imgtmp = cv2.resize(imgtmp, (w//3, h//3))
        framesall.append(imgtmp)
        # del image_origin, image, pred
    cv2.destroyAllWindows()
    with imageio.get_writer(os.path.join(abspath, r'gifman.gif'), mode="I") as obj:
        for id, frame in enumerate(framesall):
            obj.append_data(frame)

if __name__=="__main__":
    trackingsomething()
