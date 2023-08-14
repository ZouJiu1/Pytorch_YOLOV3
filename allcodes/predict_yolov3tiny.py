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
from models.Yolov3tiny import yolov3tinyNet
from PIL import Image
from config.config_yolov3tiny import *
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
    imgpath = r"/root/project/Pytorch_YOLOV3/datas/val.txt"
    savepath = r'/root/project/Pytorch_YOLOV3/images/yolovtiny'
    os.makedirs(savepath, exist_ok = True)
    # pretrainedmodel=r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\log\yolov3tiny\2023-04-15yflight\model_e2_t9624_map[0.02532625_0.0013636 ]_l0.126_2023-04-15.pth'
    # imgpath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\datas\valid\valid.txt'
    # savepath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\images\730\valid'
    for i in os.listdir(savepath):
        os.remove(os.path.join(savepath, i))
    model = yolov3tinyNet(num_classes, anchors, device, inputwidth)
    
    if torch.cuda.is_available():
        state_dict = torch.load(pretrainedmodel,map_location=torch.device('cuda'))
    else:
        state_dict = torch.load(pretrainedmodel,map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['state_dict'], strict=True)

    # pretrained = loadtorchmodel(pretrainedmodel)
    # model.load_state_dict(pretrained, strict=True)

    print('loaded', pretrainedmodel)
    lis = []
    with open(imgpath, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            lis.append(i)

    inpth = r'/root/autodl-tmp/val2017'
    lis=[os.path.join(inpth, i) for i in os.listdir(inpth)]

    np.random.seed(999)
    np.random.shuffle(lis)
    model.to(device)
    model.eval()
    score_thresh = 0.6
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
        img = TF(image)
        img = torch.unsqueeze(img, 0).to(device)
        with torch.no_grad():
            pred  = model(img)
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
