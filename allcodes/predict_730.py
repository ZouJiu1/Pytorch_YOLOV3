#Author：ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import time  
import cv2
import sys
nowpath = os.path.abspath("./")
sys.path.append(nowpath)
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from loaddata.load_datas_717_730 import trainDataset, collate_fn
# from models.Yolov3_717_730 import Yolov3
from models.Yolov3_730 import Yolov3
from PIL import Image
import torch.optim as optim
from config.config730 import *
# from tensorboardX import SummaryWriter
# from torchviz import make_dot

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# imgpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3myself\cocoval2017.txt'
# device = "cuda" if torch.cuda.is_available() else "cpu"

def evalvalid(model, ):
    pass

def predict_batch():    
    imgpath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\datas\train\train.txt'
    savepath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\images\730\train_overfit'

    # imgpath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\datas\valid\valid.txt'
    # savepath = r'C:\Users\ZouJiu\Desktop\PAT\Pytorch_YOLOV3\images\730\valid'
    for i in os.listdir(savepath):
        os.remove(os.path.join(savepath, i))
    model = Yolov3(num_classes, anchors, strides, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
    if torch.cuda.is_available():
        state_dict = torch.load(pretrainedmodel,map_location=torch.device('cuda')) 
    else:
        state_dict = torch.load(pretrainedmodel,map_location=torch.device('cpu')) 
    model.load_state_dict(state_dict['state_dict'], strict=False)
    print('loaded', pretrainedmodel)
    lis = []
    with open(imgpath, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            lis.append(i)
    np.random.seed(999)
    np.random.shuffle(lis)
    model.to(device)
    model.eval()
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
        p  = model(img, True)[0]

        #可视化  tensorboard --logdir=./
        '''
            with SummaryWriter("./log", comment="sample_model_visualization") as sw:
                sw.add_graph(model, img)
            g = make_dot(model(img))
            g.render('./log/modelviz', view=True)
            exit(0)
        '''
        if p.shape[1]==0:
            continue
        p = p[0]
        print(p.size(), p)
        cxp, cyp, wp, hp, maxscore, label = p[:,0], p[:,1], p[:,2], p[:,3], p[:,4], p[:,5]
        xmin = (cxp - wp/2)*w
        ymin = (cyp - hp/2)*h
        xmax = (cxp + wp/2)*w
        ymax = (cyp + hp/2)*h
        cvfont = cv2.FONT_HERSHEY_SIMPLEX
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for j in range(len(label)):
            minx, miny, maxx, maxy =  min(w-1, max(0, int(xmin[j]))), min(h-1, max(0, int(ymin[j]))), \
                min(w-1, max(0, int(xmax[j]))), min(h-1, max(0, int(ymax[j])))
            try:
                cv2.rectangle(image, (minx, miny), (maxx, maxy), [255, 0, 0], 2)
            except Exception as e:
                print(e)
                continue
            text = classes[int(label[j])] + ' ' + str(round(maxscore[j].item(),3))
            # text = text.replace('4','9')
            cv2.putText(image, text, (minx, miny+13), cvfont, 0.5, [128, 0, 128], 1)
            # print(classes[int(label[j])], end=' ')
        # print() 
        na = i.split(os.sep)[-1]
        if len(label)>0:
            print(ind, i)
            cv2.imwrite(r'%s/%s'%(savepath, na), image)


        # image = cv2.resize(image, (300,600))
        # cv2.imshow('img', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    predict_batch()
