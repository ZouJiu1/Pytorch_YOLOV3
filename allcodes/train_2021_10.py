#encoding=utf-8
#Author: ZouJiu
#Time: 2021-8-13

import numpy as np
import torch
import os
import time
import sys
sys.path.append(r'/home/featurize/work/Pytorch_YOLOV3')

from torch.utils.data import Dataset, DataLoader
from loaddata.load_datas_2021_10 import TF, trainDataset, collate_fn
from models.Yolov3_2021_10 import Yolov3
import torch.optim as optim
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tqdm

def adjust_lr(optimizer, stepiters, epoch):
    if stepiters < 100:
        lr = 0.000001
    elif stepiters < 200:
        lr = 0.00001
    elif stepiters < 300:
        lr = 0.0001
    elif epoch < 330:
        lr = 0.001
    elif epoch < 360:
        lr = 0.0001
    else:
        import sys
        sys.exit(0)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def trainer():
    trainpath = r'/home/featurize/work/Pytorch_YOLOV3/2023/PyTorch-YOLOv3-master/data/person/personcartrain.txt'
    traindata = trainDataset(trainpath, transform=TF)
    testdata = r'C:\Users\10696\Desktop\yolov3\datas\VOC2007\test.txt'
    pretrainedmodel = r'C:\Users\10696\Desktop\yolov3\log\model_337_881000_0.001_2021-09-03_20-36-52.pth'
    
    scratch = True
    tim = datetime.datetime.strftime(datetime.datetime.now(),"%Y-%m-%d %H-%M-%S").replace(' ', '_')
    logfile = r'log\log_%s.txt'%tim
    flogs = open(logfile, 'w')

    num_classes = 20 #voc2007_2012
    inputwidth = 416
    anchors = [[10,13], [16,30], [33,23],\
        [30,61],  [62,45],  [59,119],  \
        [116,90],  [156,198],  [373,326]]
    ignore_thresh = 0.5 #iou>0.7 confidence loss
    score_thresh = 0.45
    nms_thresh = 0.35
    num_epochs = 361
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Yolov3(num_classes, anchors, ignore_thresh, inputwidth,device,\
        score_thresh = score_thresh, nms_thresh = nms_thresh)
    print(model)
    flogs.write(str(model)+'\n')
    if not os.path.exists(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    if pretrainedmodel and os.path.exists(pretrainedmodel):
        print('loading pretrained model: ', pretrainedmodel)
        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location='cuda')
        else:
            state_dict = torch.load(pretrainedmodel, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
        if not scratch:
            iteration = state_dict['iteration']
            alliters = state_dict['alliters']
            nowepoch = state_dict['nowepoch']
        else:
            iteration = 0
            alliters = 0
            nowepoch = 0
        print('loading complete')
    else:
        print('no pretrained model')
        iteration = 0
        alliters = 0
        nowepoch = 0
    model = model.to(device)

    adam = False
    lr = 0.001 # initial learning rate (SGD=1E-2, Adam=1E-3)
    momnetum=0.937
    batch_size = 200
    subsiz = 10
    params = [p for p in model.parameters() if p.requires_grad]
    if adam:
        optimizer = optim.Adam(params, lr=lr, betas=(momnetum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(params, lr=lr, momentum=momnetum, nesterov=True)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=7,
    #                                                gamma=0.1)
    torch.manual_seed(999999)
    dataloader = DataLoader(traindata, batch_size=batch_size//subsiz,shuffle=True, \
        num_workers=2,collate_fn=collate_fn)
    start = time.time()
    print('Using {} device'.format(device))
    flogs.write('Using {} device'.format(device)+'\n')
    stepiters = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
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
        losscol = []
        optimizer.zero_grad()
        # for i, (image, label) in enumerate(dataloader):
        for i, (image, label) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):
            stepiters += 1
            if stepiters<alliters:
                continue
            count += 1
            adjust_lr(optimizer, stepiters, epoch) #
            image = image.to(device)
            label = label.to(device)
            result3, result2, result1, objectness, recall50, recall75, noobjectness, lossiou = model(image, label)
            loss = result3 + result2 + result1
            loss.requires_grad_(True)
            loss = loss.to(device)
            loss.backward()
            losscol.append(loss.detach().cpu().item())
            # statistics
            running_loss += loss.item()
            epoch_loss = running_loss / count
            logword = '\nepoch: {}, iteration: {}, alliters: {}, lr: {}, objectness: {:.3f}, noobjectness: {:.6f}, iou: {:.3f}, recall50: {:.3f}, recall75: {:.3f}, loss: {:.3f}, avgloss: {:.3f}'.format(
                epoch, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], objectness, noobjectness, lossiou, \
                    recall50, recall75, loss.item(), epoch_loss)
            if i%subsiz==0 or i == len(dataloader)-1:
                optimizer.step()
                optimizer.zero_grad()
                print(logword)
                flogs.write(logword+'\n')
                flogs.flush()
            savestate = {'state_dict':model.state_dict(),\
                        'iteration':i,\
                        'alliters':stepiters,\
                        'nowepoch':epoch}
            if stepiters%500==0 and count!=1:
                try:
                    torch.save(savestate, r'C:\Users\10696\Desktop\yolov3\log\model_{}_{}_{:.3f}_{}.pth'.format(epoch, stepiters, loss.item(),tim))
                except:
                    pass
        scheduler.step(np.mean(losscol))
        # lr_scheduler.step()
        iteration=0
        try:
            torch.save(savestate, r'C:\Users\10696\Desktop\yolov3\log\model_{}_{}_{:.3f}_{}.pth'.format(epoch, stepiters, loss.item(),tim))
        except:
            pass
        # evaluate(model, dataloader_test, device = device)
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()


if __name__ == '__main__':
    trainer()
