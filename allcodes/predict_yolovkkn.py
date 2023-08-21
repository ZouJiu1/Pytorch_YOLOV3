#Author：ZouJiu
#Time: 2023-7-28
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import numpy as np
import torch
import time
import cv2
from models.Yolovkkn import YolovKKNet
from PIL import Image
from config.config_yolovKKn import *
from utils.utils_yolov3tiny import non_max_suppression, loadtorchmodel, scale_coords
# from tensorboardX import SummaryWriter
# from torchviz import make_dot

# imgpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3myself\cocoval2017.txt'
# device = "cuda" if torch.cuda.is_available() else "cpu"

def evalvalid(model, ):
    pass

def predict_batch():
    # imgpath = r'/home/featurize/work/Pytorch_YOLOV3/datas/val.txt'
    # imgpath = r'/home/featurize/work/Pytorch_YOLOV3/2023/PyTorch-YOLOv3-master/data/person/personcartrain.txt'
    imgpath = r"C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\train.txt"
    savepath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\images\yolokkn201#'
    os.makedirs(savepath, exist_ok = True)
    # pretrainedmodel = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\log\yolovkkn\2023-08-16yolokkn\model_e51_map[0.48195__0.256185]_l64988.657_2023-08-16.pth'
    pretrainedmodel = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\log\yolovkkn\2023-08-18yolokkn\model_e30_map[0.396382__0.245314]_l8.850_2023-08-18.pt'
    # imgpath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\datas\valid\valid.txt'
    # savepath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\images\730\valid'
    for i in os.listdir(savepath):
        os.remove(os.path.join(savepath, i))
    model = YolovKKNet(num_classes, anchors, device, inputwidth)
    
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
    yolovfive = True if chooseLoss in ["20230730", "yolofive"] else False
    # pretrained = loadtorchmodel(pretrainedmodel)
    # model.load_state_dict(pretrained, strict=True)
    for i in range(len(model.yolo)):
        model.yolo[i].eval()
        model.yolo[i].device = device
    print('loaded', pretrainedmodel)
    lis = []
    with open(imgpath, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            lis.append(i)

    # inpth = r'F:\val201seven'
    # inpth = r'C:\Users\10696\Desktop\kkk'
    inpth = r'F:\COCOIMAGE\coco\images\val201#'
    lis = [os.path.join(inpth, i) for i in os.listdir(inpth)]

    np.random.seed(999)
    np.random.shuffle(lis)
    model.to(device)
    model.eval()
    score_thresh = 0.38
    nms_thresh = 0.6 - 0.2 - 0.1
    # np.random.shuffle(lis)
    for ind, i in enumerate(lis):
        # img = cv2.imread(i)
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # h,w,c=img.shape
        # img = cv2.resize(image, (416, 416))
        # img = img/255
        # img = (img-0.5)/0.5
        # img = np.expand_dims(img, 0)
        # img = np.transpose(img, (0, 3, 1, 2))
        # img = torch.FloatTensor(img)

        image = Image.open(i).convert("RGB")
        w, h = image.size
        img = TFRESIZE(image)
        img = torch.unsqueeze(img, 0).to(device)
        with torch.no_grad():
            pred  = model(img, yolovfive=yolovfive)
            # print(pred[...])
            pred = non_max_suppression(pred, score_thresh, nms_thresh, agnostic=False)
            #可视化  tensorboard --logdir=./
            '''
                with SummaryWriter("./log", comment="sample_model_visualization") as sw:
                    sw.add_graph(model, img)
                g = make_dot(model(img))
                g.render('./log/modelviz', view=True)
                exit(0)
            '''
            print(pred)
            cvfont = cv2.FONT_HERSHEY_SIMPLEX
            image = np.asarray(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cnt = 0
            for j, det in enumerate(pred):
                # Rescale boxes from img_size to im0 size
                if len(det)==0:
                    continue
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                for *xyxy, conf, label in reversed(det):
                    xmin, ymin, xmax, ymax = xyxy
                    xmin, ymin, xmax, ymax = xmin.cpu().item(), ymin.cpu().item(), xmax.cpu().item(), ymax.cpu().item()
                    conf = conf.cpu().item()
                    label = label.cpu().item()
                    minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin))), min(h-1, max(0, int(ymin))), \
                        min(w-1, max(0, int(xmax))), min(h-1, max(0, int(ymax)))
                    areak = (maxx - minx) * (maxy - miny)
                    kk = np.prod(image.shape[:-1])
                    k = areak / kk
                    if k > (1-0.1) and len(det) >= 2:
                        continue
                    try:
                        cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 2)
                        cnt += 1
                    except Exception as e:
                        print(e)
                        continue
                    text = classes[int(label)] + ' ' + str(round(conf,3))
                    # text = text.replace('4','9')
                    cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [255, 0, 0], 1)
                    # print(classes[int(label[j])], end=' ')
                # print() 
            na = i.split(os.sep)[-1]
            if cnt > 0:
                print(ind, i)
                cv2.imwrite(r'%s/%s'%(savepath, na), image)

            # image = cv2.resize(image, (300,600))
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
if __name__ == '__main__':
    predict_batch()
