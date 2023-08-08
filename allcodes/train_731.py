#encoding=utf-8
#Author: ZouJiu
#Time: 2021-8-13


import os
import cv2
import sys
nowpath = os.path.abspath("./")
sys.path.append(nowpath)
import time
import torch
import datetime
import numpy as np
# from Yolov3copy23 import Yolov3
from models.Yolov3_731 import lossyolo
import sys
sys.path.append(r"/home/Pytorch_YOLOV3")
from models.yolov3 import Yolov3
import torch.optim as optim
from utils.utils_731 import load_darknet_weights, intialization, freeze_darknet_backbone
from utils.common import cvshow, validDataset, collate_fn
from utils.validation import validation_map
from torch.utils.data import Dataset, DataLoader
from loaddata.load_datas_731 import trainDataset
from config.config import *

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def adjust_lr(optimizer, stepiters, epoch, Adam, freeze_backbone, \
              momnetum, learning_rate, model, weight_decay, flogs):
    steps0 = 100
    steps1 = 200
    steps2 = 300
    if stepiters < steps0:
        lr = (0.001 - 0.0001) * stepiters/steps0 + 0.0001
    elif stepiters < steps1:
        lr = (0.0001 - 0.00001) * stepiters/steps1 + 0.00001
    elif stepiters < steps2:
        lr = (0.001 - 0.0001) * stepiters/steps2 + 0.0001
    elif epoch < 73:
        lr = 0.001
    elif epoch == 73: #unfreeze
        for p in model.parameters():
            p.requires_grad = True

        params = [p for p in model.parameters() if p.requires_grad]
        print('unfreeze all layer to train, now trainable layer number is : ', len(params))
        flogs.write('unfreeze all layer to train, now trainable layer number is : %d\n'%len(params)+'\n')
        if Adam:
            optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
        lr = 0.0001
    elif epoch < 121:
        lr = 0.0001
    elif epoch < 130:
        lr = 0.00001
    elif epoch < 136:
        lr = 0.000001
    else:
        import sys
        sys.exit(0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def trainer():
    #pip3 install --user --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    traindata = trainDataset(trainpath, stride = strides, anchors = anchors, anchor_per_layer = anchor_per_layer,\
                             device=device, inputwidth = inputwidth, numclasses = num_classes,transform=TF)
    validdata = validDataset(validpath, transform=TF)
    flogs = open(logfile, 'w')
    # model = Yolov3(num_classes, anchors, strides, ignore_thresh, inputwidth,device,\
    #     score_thresh = score_thresh, nms_thresh = nms_thresh)
    model = Yolov3(num_classes, anchors, strides, \
        ignore_thresh, inputwidth,device, anchor_per_layer)
    lossfun = lossyolo(iou_thresh, ignore_thresh, strides, device, inputwidth)
    print(model)
    flogs.write(str(model)+'\n')
    iteration = 0
    alliters = 0
    nowepoch = 0
    if intialze:
        intialization(model)
    if not os.path.exists(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    if pretrainedmodel and os.path.exists(pretrainedmodel) and not load_darknet_w:
        print('loading pretrained model: ', pretrainedmodel)
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location='cuda')
        else:
            state_dict = torch.load(pretrainedmodel, map_location='cpu')
        delete = []
        for key, valu in state_dict['state_dict'].items():
            if 'hb3.conv7' in key or 'hb2.conv7' in key or 'hb1.norm7' in key or 'hb1.conv7' in key or 'hb2.norm7' in key or 'hb3.norm7' in key:
                delete.append(key)
        for key in delete:
            state_dict['state_dict'].pop(key)
        # exit(0)
        model.load_state_dict(state_dict['state_dict'], strict = False)
        if not scratch:
            iteration = state_dict['iteration']
            alliters = state_dict['alliters']
            nowepoch = state_dict['nowepoch']
        print('loading complete')
    elif not os.path.exists(pretrainedmodel) and not load_darknet_w:
        print('file not found, there is no pretrained model, train from scratch')

    # if load_darknet_w:
        # load_darknet_weights(model, r"C:\Users\ZouJiu\Desktop\projects\tmp\darknet53_448.weights") #r"log\darknet53_448.weights")
        # load_darknet_weights(model, r"/home/Pytorch_YOLOV3\log\darknet53.conv.74")
        # load_darknet_weights(model, r"/home/Pytorch_YOLOV3\log\yolov3.weights")
        # print('loaded darknet weight......')
    model = model.to(device)
    lossfun = lossfun.to(device)

    model.load_darknet_weights(r"/home/Pytorch_YOLOV3\log\yolov3.weights")
    if freeze_backbone:       #step1 freeze darknet53 backbone parameters to train, because you data number is small
        bre = len([p for p in model.parameters() if p.requires_grad])
        freeze_darknet_backbone(model)
        print('before freeze trainable layer number is: ', bre)
        flogs.write('before freeze trainable layer number is: %d\n'%bre)
        bre = len([p for p in model.parameters() if p.requires_grad])
        print('after freeze trainable layer number is: ', bre)
        flogs.write('after freeze trainable layer number is: %d\n'%bre)

    params = [p for p in model.parameters() if p.requires_grad]
    if Adam:
        optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=7,
    #                                                gamma=0.1)
    torch.manual_seed(time.time())
    dataloader = DataLoader(traindata, batch_size=batch_size,shuffle=True, \
        num_workers=2)  #,collate_fn=collate_fn)
    validloader = DataLoader(validdata, batch_size=1,shuffle=True, num_workers=1)
    start = time.time()
    print('Using {} device'.format(device))
    length = len(dataloader)
    flogs.write('Using {} device'.format(device)+'\n')
    stepiters = 0
    pre_map = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        flogs.write('Epoch {}/{}'.format(epoch, num_epochs)+'\n')
        print('-'*10)
        running_loss = 0
        if epoch<nowepoch:
            stepiters += len(dataloader)
            continue
        model.train()
        count = 0
        for i, (image, label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox) in enumerate(dataloader):
            # cvshow(image, label)   #cv2 show inputs images
            stepiters += 1
            if stepiters<alliters:
                continue
            count += 1
            optimizer = adjust_lr(optimizer, stepiters, epoch, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay, flogs)
            
            image = image.to(device)
            p, p_d = model(image)
            loss, loss_giou, loss_conf, loss_cls, recall50, recall75, obj, noobj = \
                lossfun(label_sbbox, label_mbbox, label_lbbox, sbbox, mbbox, lbbox, \
                           p, p_d)
            # loss.requires_grad_(True)
            # loss = loss.to(device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_loss = running_loss / count
            logword = '''epoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.6f}, obj: {:.3f}, noobj: {:.6f}, \
recall50: {:.3f}, recall75: {:.3f}, loss: {:.3f}, avgloss: {:.3f}, loss_giou: {:.3f}, loss_conf: {:.3f}, loss_cls: {:.3f}'''.format(
                       epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], obj, noobj, \
                       recall50, recall75, loss.item(), epoch_loss, loss_giou, loss_conf, loss_cls)
            print(logword)
            flogs.write(logword+'\n')
            flogs.flush()
            savestate = {'state_dict':model.state_dict(),\
                        'iteration':i,\
                        'alliters':stepiters,\
                        'nowepoch':epoch}
        map = validation_map(model, validloader)
        model.train()
        print("validation......num_img: {}, mAP: {}, premap:{}".format(len(validdata), map, pre_map))
        try:
            if(pre_map < map) or epoch%3==2:
                torch.save(savestate, r'/home/Pytorch_YOLOV3\log\model_{}_{}_map{}_{:.3f}_{}.pth'.format(epoch, stepiters, map, loss.item(),tim))
            print('savemodel ')
        except:
            print('error: don\'t savemodel')
        # evaluate(model, dataloader_test, device = device)
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()

if __name__ == '__main__':
    trainer()
