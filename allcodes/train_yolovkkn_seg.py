#encoding=utf-8
#Author：ZouJiu
#Time: 2022-12-10

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import cv2
import time
import torch
import datetime
import numpy as np
from copy import deepcopy
from models.Yolovkkn_seg import YolovKKNet
# from models.layer_loss_20230816 import calculate_losses_darknet, calculate_losses_Alexeydarknet, calculate_losses_yolofive, \
from models.layer_loss_segment import calculate_losses_darknet, calculate_losses_Alexeydarknet, calculate_losses_yolofive, \
    calculate_losses_darknetRevise, calculate_losses_20230730, calculate_losses_yolofive_revise
# , calculate_losses_yolofive_original
import torch.optim as optim
from utils.common import cvshow_seg, collate_fnseg, provide_determinism, smart_optimizer, ModelEMA, de_parallel
from utils.validation_yolov3tiny import validation_map_seg
from torch.utils.data import Dataset, DataLoader
# from loaddata.load_datas_yolov3tiny import trainDataset
from loaddata.cocoread_seg import trainDataset
from config.config_yolovKKn import *
from multiprocessing import cpu_count
from utils.evaluate_yolov3tiny import evaluation
import math
import onnxsim
import onnx
import tqdm
# torch.autograd.set_detect_anomaly(True)

def adjust_lr(optimizer, stepiters, epoch, num_batch, num_epochs, batch_size, \
              momnetum, learning_rate):
    steps0 = num_batch * warmepoch
    final_lr = 0.01
    baselr = learning_rate
    nbk = 2**6  # nominal batch size
    lf = lambda k : 1 + ((final_lr - 1) / (num_epochs - 1)) * (k - 1)
    if epoch <= warmepoch:
        # lr = baselr * np.exp(stepiters * 6 / steps0 - 6)
        lr = (baselr / steps0) * stepiters
        xi = [0, steps0]
        ni = stepiters
        #https://github.com/ultralytics/yolov5/blob/master/train.py#L301
        # accumulate = max(1, np.interp(ni, xi, [1, nbk / batch_size]).round())
        for j, x in enumerate(optimizer.param_groups):
            # x['lr'] = (0.01 / (len(train_loader) * 3)) * steps
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            warmup_bias_lr = baselr * 10
            x['lr'] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, baselr * lf(epoch)])
            if 'momentum' in x:
                x['momentum'] = np.interp(ni, xi, [warmup_momnetum, momnetum])
        return optimizer.param_groups[0]['lr']

    # elif epoch < (num_epochs*(8/10)):
    #     lr = baselr
    # elif epoch < (num_epochs*(9/10)):
    #     lr = baselr*1e-1
    # else:
    #     lr = baselr*1e-2
    else:
        lr = ((1 - np.cos(epoch * np.pi / num_epochs)) / 2) * (final_lr - 1) + 1
        lr = baselr * lr
    # else:
    #     lr = 1 + ((final_lr - 1) / (num_epochs - 1)) * (epoch - 1)
    #     lr = baselr * lr

    # k = []
    # for epoch in range(1, num_epochs+1):
    #     lr = 1 + ((final_lr - 1) / (num_epochs - 1)) * (epoch - 1)
    #     lr = baselr * lr
    #     k.append(lr)
    # kk = 0

    # elif epoch < num_epochs*2//3:
    #     lr = baselr
    # elif epoch < num_epochs * 9//10:
    #     lr = baselr*1e-1
    # elif epoch < num_epochs:
    #     lr = baselr*1e-2
    # else:
    #     import sys
    #     sys.exit(0)
    # if epoch == 10: #unfreeze
    #     for p in model.parameters():
    #         p.requires_grad = True

    #     params = [p for p in model.parameters() if p.requires_grad]
    #     print('unfreeze all layer to train, now trainable layer number is : ', len(params))
    #     flogs.write('unfreeze all layer to train, now trainable layer number is : %d\n'%len(params)+'\n')
    #     if Adam:
    #         optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    #     else:
    #         optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
    #     lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def trainer():
    seed = 612387967
    provide_determinism(seed)
    torch.cuda.manual_seed_all(612387967)
    torch.manual_seed(612387967)
    #pip3 install --user --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    traindata = trainDataset(trainpath, train_imgpath, stride = strides, anchors = anchors, \
                             augment = True, inputwidth = inputwidth, transform=TFRESIZE)
    count_scale = traindata.count_scale.to(device)

    flogs = open(logfile, 'w')
    model = YolovKKNet(num_classes, anchors, device, inputwidth)
    # yolov3 = Yolov3().to(self.device)
    print(model)
    flogs.write(str(model)+'\n')
    iteration = 0
    alliters = 0
    nowepoch = 0
    # if intialze:
    #     intialization(model)
    if not os.path.exists(pretrainedmodel) or os.path.isdir(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    elif pretrainedmodel and os.path.exists(pretrainedmodel) and not load_darknet_w:
        print('loading pretrained model: ', pretrainedmodel)
        
        # pretrained = loadtorchmodel(pretrainedmodel)
        # model.load_state_dict(pretrained, strict = False)
        # del pretrained

        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cpu'))
        if pretrainedmodel.endswith("pt"):
            param = state_dict['state_dict'].float()
            param = param.state_dict()
        else:
            param = state_dict['state_dict']
        kkk = {}
        for key, value in param.items():
            if 'out0' in key or 'out1' in key or 'out2' in key:
                continue
            kkk[key.replace("module.", "")] = value
        pretrained = kkk
        model.load_state_dict(pretrained, strict = False)
        if not scratch and 'iteration' in state_dict.keys():
            iteration = state_dict['iteration']
            alliters = state_dict['alliters']
            nowepoch = state_dict['nowepoch']
        del state_dict
        print('loading complete')
    elif not os.path.exists(pretrainedmodel) and not load_darknet_w:
        print('file not found, there is no pretrained model, train from scratch')

    # alliters = 0
    # nowepoch = 0

    # if load_darknet_w:
    #     # load_darknet_weights(model, r"C:\Users\ZouJiu\Desktop\projects\tmp\darknet53_448.weights") #r"log\darknet53_448.weights")
    #     # load_darknet_weights(model, r"/home/Pytorch_YOLOV3\log\darknet53.conv.74")
    #     load_darknet_weights(model, darknet_weight)
    #     print('loaded darknet weight......')

    model = model.to(device)

    # if freeze_backbone:       #step1 freeze darknet53 backbone parameters to train, because you data number is small
    #     bre = len([p for p in model.parameters() if p.requires_grad])
    #     freeze_darknet_backbone(model)
    #     print('before freeze trainable layer number is: ', bre)
    #     flogs.write('before freeze trainable layer number is: %d\n'%bre)
    #     bre = len([p for p in model.parameters() if p.requires_grad])
    #     print('after freeze trainable layer number is: ', bre)
    #     flogs.write('after freeze trainable layer number is: %d\n'%bre)

    # params = [p for p in model.parameters() if p.requires_grad]
    # if Adam:
    #     optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    # else:
    ema = ModelEMA(model)
    optimizer = smart_optimizer(model, 'SGD', lr = learning_rate, momentum=momnetum, decay=weight_decay)
    # k = model.parameters()
    # optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
    # optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=7,
    #                                                gamma=0.1)
    nc = 11 #cpu_count()
    num_cpu = min([nc//max(torch.cuda.device_count(), 1), 2**3])  #num_cpu if num_cpu < 20 else 13
    dataloader = DataLoader(traindata, batch_size = batch_size//subsiz,shuffle=True, \
        num_workers=num_cpu, collate_fn=collate_fnseg, pin_memory=True)
    valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                augment = False, inputwidth = inputwidth, transform=TFRESIZE)
    valdataloader = DataLoader(valdata, batch_size = batch_size // subsiz, shuffle=False, \
            num_workers=num_cpu, collate_fn=collate_fnseg, pin_memory=True)
    # validloader = DataLoader(validdata, batch_size=1,shuffle=True, num_workers=1)
    start = time.time()
    print('Using {} device'.format(device))
    length = len(dataloader)
    flogs.write('Using {} device'.format(device)+'\n')
    stepiters = 0
    pre_map = 0

    bce0loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(device)
    bce1loss = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    bce2loss = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    bcecls = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    bcecof = torch.nn.BCEWithLogitsLoss(reduction='sum').to(device)
    mseloss = [torch.nn.MSELoss(reduction='sum').to(device) for i in range(2*2)]
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    yolovfive = True if chooseLoss in ["20230730", "yolofive"] else False
    amp = False
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        flogs.write('Epoch {}/{}'.format(epoch, num_epochs)+'\n')
        print('-'*10)
        running_loss = 0
        if epoch<=nowepoch:
            stepiters += len(dataloader)
            continue
        count = 0
        model.train()
        losscol = []
        optimizer.zero_grad()

        lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, batch_size, momnetum, learning_rate)
        # if epoch <= warmepoch:
        #     optimizer.momentum = warmup_momnetum
        # elif epoch == warmepoch + 1:
        #     optimizer.momentum = momnetum
        preiou = 0
        for i, (image, labels, gtmask, imgid) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):
            stepiters += 1
            # if stepiters < alliters:
            #     continue
            # cvshow_seg(image, labels, gtmask)   #cv2 show inputs images)
            count += 1
            if epoch <= warmepoch:
                lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, batch_size, momnetum, learning_rate)
            
            image = image.to(device)
            labels = labels.to(device)
            gtmask = gtmask.to(device)
            
            # try:
            with torch.cuda.amp.autocast(amp):
                # try:
                prediction = model(image, yolovfive = yolovfive)
                # except Exception as e:
                #     continue

                # loss, c_l, confi_l, iouloss = calculate_losses_yolov3(prediction, labels, model, count_scale)
                if chooseLoss == "darknetRevise":
                    loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_darknetRevise(prediction, labels, model, ignore_thresh, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)
                elif chooseLoss == "darknet":
                    loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_darknet(prediction, labels, model, ignore_thresh, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)
                elif chooseLoss == "Alexeydarknet":
                    loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_Alexeydarknet(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)
                elif chooseLoss == "yolofive":
                    loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_yolofive(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)
                        # loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_yolofive_original(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                        #                                                                     bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)
                elif chooseLoss == "20230730":
                    loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, boxnum = calculate_losses_20230730(prediction, labels, model, count_scale, ignore_thresh, \
                                                                                        bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask)

                # if darknetLoss and i > 30 and iouloss > 10 and iouloss / preiou > 3:
                #     loss = c_l + confi_l

                if amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()                
                # loss, loss_components = computeloss(prediction, labels, device, model)
                losscol.append(loss.detach().cpu().item())
                
                # if torch.isnan(loss).item()==False:
                loss = loss.detach().cpu().item()
                running_loss += loss
                preiou = iouloss
                # else:
                #     optimizer = adjust_lr(optimizer, 200, epoch, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay, flogs)
                #     print(torch.isnan(loss).item(), torch.isnan(loss).item()==False)
                # statistics
                epoch_loss = running_loss / count
    #             logword = '''\nepoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.6f}, MSE loss: {:.6f}, Class loss: {:.3f}, \
    # Confi loss: {:.3f}, iouloss: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}'''.format(
    #                                    epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], float(mse.item()),\
    #                                     float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss)
                logword = '''\ne: {}, r:{:.2f}%, i: {}, ai: {}, lr: {:.6f}, Class: {:.3f}, \
    Confi: {:.3f}, iou: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}, iounow: {:.3f}, cof: {:.3f}, ncof: {:.6f}, cla: {:.3f}, mask: {:.3f}'''.format(
                                    epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], \
                                        float(c_l.item()/boxnum), float(confi_l.item()/boxnum), iouloss.item()/boxnum, loss/boxnum, \
                                            epoch_loss/boxnum, iounow.item(), cof.item(), ncof.item(), cla.item(), maskloss.item())
                if i%subsiz==0 or i == len(dataloader)-1:
                    if amp:
                        scaler.unscale_(optimizer)  # unscale gradients
                    # https://jamesmccaffrey.wordpress.com/2022/10/17/the-difference-between-pytorch-clip_grad_value_-and-clip_grad_norm_-functions/
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                    if amp:
                        scaler.step(optimizer)  # optimizer.step
                        scaler.update()
                    else:
                        optimizer.step() #C:\Users\10696\Desktop\Pytorch_YOLOV3\\datas\train\images\2010_003635.jpg
                    optimizer.zero_grad()
                    ema.update(model)

                    print(logword)
                    flogs.write(logword+'\n')
                    flogs.flush()
        __savepath__ = os.path.join(savepath, datekkk) + prefix
        os.makedirs(__savepath__, exist_ok=True)

        # onnxfil = __savepath__+os.sep+r'model_e{}_t{}_l{:.3f}_{}.onnx'.format(epoch, stepiters, epoch_loss, datekkk)
        # model.eval()
        # torch.onnx.export(
        #     model.cpu(),  # --dynamic only compatible with cpu
        #     image[0].unsqueeze(0).cpu(),
        #     onnxfil,
        #     verbose=False,
        #     # opset_version=opset,
        #     do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        #     input_names=['images'],
        #     output_names=['output0'])
        # model_onnx = onnx.load(onnxfil)
        
        # model_onnx, check = onnxsim.simplify(model_onnx)
        # assert check, 'assert check failed'
        # onnx.save(model_onnx, onnxfil)

        model = model.to(device)
        # ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        savestate = {'state_dict': deepcopy(de_parallel(model)).half(), \
                     'ema': deepcopy(ema.ema).half(), \
                     'iteration':i,\
                     'alliters':stepiters,\
                     'nowepoch':epoch}
        # scheduler.step(np.mean(losscol))
        [map, mAP0], lengthkk = validation_map_seg(model if ema==None else ema.ema, yolovfive, valdataloader, device)
        lengthkk = 2000*2+1000
        # map = evaluation("", model=model, dataloader=valdataloader, score_thresh_now = 0.001, nms_thresh_now = 0.6)
        print("validation......num_img: {}, mAP: {}, premap:{}".format(lengthkk, [map, mAP0], pre_map))
        if len(map) > 2:
            map = [round(map[0], 6), round(np.mean(map), 6)]
        strmap = str(map).replace(",", "_").replace(" ", "_")

        torch.save(savestate, __savepath__+os.sep+r'model_e{}_t{}_map{}_l{:.3f}_{}.pt'.format(epoch, stepiters, strmap, epoch_loss, datekkk))
        if(pre_map < np.mean(map)) or (epoch+1)%1==0 or epoch==num_epochs-1:
            torch.save(savestate, __savepath__+os.sep+r'model_e{}_t{}_map{}_l{:.3f}_{}.pt'.format(epoch, stepiters, strmap, epoch_loss, datekkk))
            print('savemodel ')
            pre_map = np.mean(map)
        del savestate
        # except:
        #     print('error: don\'t savemodel')
        # evaluate(model, dataloader_test, device = device)
    
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()

if __name__ == '__main__':
    trainer()
