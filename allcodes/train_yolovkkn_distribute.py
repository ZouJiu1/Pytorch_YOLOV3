#encoding=utf-8
#Author：ZouJiu
#Time: 2023-7-27

import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

from config.config_yolovKKn import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

import time
import torch
import numpy as np
from copy import deepcopy
from models.Yolovkkn import YolovKKNet
# from models.layer_loss_20230816 import calculate_losses_darknet, calculate_losses_Alexeydarknet, calculate_losses_yolofive, \
from models.layer_loss import calculate_losses_darknet, calculate_losses_Alexeydarknet, calculate_losses_yolofive, \
    calculate_losses_darknetRevise, calculate_losses_20230730, calculate_losses_yolofive_revise
#, calculate_losses_yolofive_original    layer_loss_20230816
import torch.optim as optim
from utils.common import cvshow_, collate_fn, provide_determinism, smart_optimizer, ModelEMA, de_parallel
from utils.validation_yolov3tiny import validation_map
from torch.utils.data import Dataset, DataLoader
from loaddata.cocoread import trainDataset
from multiprocessing import cpu_count
import torch.distributed as dist
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch import nn
import tqdm
import torch.multiprocessing as mp
# # torch.autograd.set_detect_anomaly(True)

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
    # else:
    #     lr = ((1 - np.cos(epoch * np.pi / num_epochs)) / 2) * (final_lr - 1) + 1
    #     lr = baselr * lr
    else:
        lr = 1 + ((final_lr - 1) / (num_epochs - 1)) * (epoch - 1)
        lr = baselr * lr

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
    '''
    apt-get update && apt-get install -y net-tools
    ifconfig
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=INFO
    NCCL_SOCKET_IFNAME=eth1  # 网卡名称更换您自己的
    NCCL_DEBUG=WARN
    python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=666 allcodes/train_yolovkkn_distribute.py
    python -m torch.distributed.launch --nproc_per_node=6 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=666 allcodes/train_yolovkkn_distribute.py
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=666 allcodes/train_yolovkkn_distribute.py
    e: \d\d, r:99.
    '''
    # https://pytorch.org/docs/stable/distributed.html#launch-utility
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    os.system("export NCCL_SOCKET_IFNAME=eth0")
    # os.system("export TORCH_DISTRIBUTED_DEBUG=DETAIL") #DEBUG
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    cudanum = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert cudanum == torch.cuda.device_count(), (cudanum, torch.cuda.device_count())
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    # exit(0)
    seed = 612387967
    provide_determinism(seed)
    torch.cuda.manual_seed_all(612387967)
    torch.manual_seed(612387967)
    
    torch.cuda.set_device(rank % torch.cuda.device_count())
    devicenow = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    #pip3 install --user --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    traindata = trainDataset(trainpath, train_imgpath, stride = strides, anchors = anchors, \
                            augment = True, inputwidth = inputwidth, transform=TF)
    num_cpu = (72 - 36)//2     # cpu_count()
    # num_cpu = min([nc//max(torch.cuda.device_count(), 1), 2**3])  #num_cpu if num_cpu < 20 else 13
    count_scale = traindata.count_scale.to(devicenow)
    
    train_sampler = DistributedSampler(dataset=traindata, shuffle=True)
    dataloader = DataLoader(traindata, batch_size = batch_size, shuffle = (train_sampler is None), \
        num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)

    if rank==0:
        valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                    augment = False, inputwidth = inputwidth, transform=TFRESIZE)
        # val_sampler = DistributedSampler(dataset=valdata, shuffle=True)
        valdataloader = DataLoader(valdata, batch_size = batch_size, shuffle=False, \
                    num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True) #, sampler=val_sampler)

    if rank == 0:
        flogs = open(logfile, 'w')
    model = YolovKKNet(num_classes, anchors, devicenow, inputwidth)
    # yolov3 = Yolov3().to(self.devicenow)
    # print(model)
    if rank == 0:
        flogs.write(str(model)+'\n')
    iteration = 0
    alliters = 0
    nowepoch = 0
    # if intialze:
    #     intialization(model)
    if not os.path.exists(pretrainedmodel) or os.path.isdir(pretrainedmodel):
        print('the pretrainedmodel do not exists %s'%pretrainedmodel)
    elif pretrainedmodel and os.path.exists(pretrainedmodel) and not load_darknet_w:
        if rank == 0:
            print('loading pretrained model: ', pretrainedmodel)
        
        # pretrained = loadtorchmodel(pretrainedmodel)
        # model.load_state_dict(pretrained, strict = False)
        # del pretrained

        if torch.cuda.is_available():
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cuda'))
        else:
            state_dict = torch.load(pretrainedmodel, map_location=torch.device('cpu'))
        kkk = {}
        for key, value in state_dict['state_dict'].items():
            kkk[key.replace("module.", "")] = value
        state_dict['state_dict'] = kkk
        pretrained = state_dict['state_dict']
        model.load_state_dict(pretrained, strict = True)
        if not scratch and 'iteration' in state_dict.keys():
            iteration = state_dict['iteration']
            alliters = state_dict['alliters']
            nowepoch = state_dict['nowepoch']
        del state_dict
        if rank == 0:
            print('loading complete')
    
    elif not os.path.exists(pretrainedmodel) and not load_darknet_w:
        if rank == 0:
            print('file not found, there is no pretrained model, train from scratch')

    # alliters = 0
    # nowepoch = 0

    # if load_darknet_w:
    #     # load_darknet_weights(model, r"C:\Users\ZouJiu\Desktop\projects\tmp\darknet53_448.weights") #r"log\darknet53_448.weights")
    #     # load_darknet_weights(model, r"/home/Pytorch_YOLOV3\log\darknet53.conv.74")
    #     load_darknet_weights(model, darknet_weight)
    #     print('loaded darknet weight......')

    # Convert BatchNorm to SyncBatchNorm. 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(devicenow)
    if rank == 0:
        ema = ModelEMA(model)
    else:
        ema = None
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # params = model.parameters() # [p for p in model.parameters() if p.requires_grad]
    # if Adam:
    #     optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    # else:
    optimizer = smart_optimizer(model, 'SGD', lr = learning_rate, momentum=momnetum, decay=weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)

    # if freeze_backbone:       #step1 freeze darknet53 backbone parameters to train, because you data number is small
    #     bre = len([p for p in model.parameters() if p.requires_grad])
    #     freeze_darknet_backbone(model)
    #     print('before freeze trainable layer number is: ', bre)
    #     flogs.write('before freeze trainable layer number is: %d\n'%bre)
    #     bre = len([p for p in model.parameters() if p.requires_grad])
    #     print('after freeze trainable layer number is: ', bre)
    #     flogs.write('after freeze trainable layer number is: %d\n'%bre)

    # and a learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=7,
    #                                                gamma=0.1)
    # validloader = DataLoader(validdata, batch_size=1,shuffle=True, num_workers=1)
    start = time.time()
    if rank == 0:
        print('Using {} device'.format(devicenow))
    length = len(dataloader)
    if rank == 0:
        flogs.write('Using {} device'.format(devicenow)+'\n')
    stepiters = 0
    pre_map = 0

    bce0loss = torch.nn.BCEWithLogitsLoss(reduction='none').to(devicenow)
    bce1loss = torch.nn.BCEWithLogitsLoss(reduction='sum').to(devicenow)
    bce2loss = torch.nn.BCEWithLogitsLoss(reduction='sum').to(devicenow)
    bcecls = torch.nn.BCEWithLogitsLoss(reduction='sum').to(devicenow)
    bcecof = torch.nn.BCEWithLogitsLoss(reduction='sum').to(devicenow)
    mseloss = [torch.nn.MSELoss(reduction='sum').to(device) for i in range(2*2)]
    epoch_loss = 0

    amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
# '''\nepoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.3f}, MSE loss: {:.3f}, Class loss: {:.3f}, \
# # Confi loss: {:.3f}, iouloss: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}'''
    yolovfive = True if chooseLoss in ["20230730", "yolofive"] else False
    nb = len(dataloader)  # number of batches
    for epoch in range(1, num_epochs + 1):
        if rank == 0:
            print(('\n' + '%9s' * (6 + 6 + 2)) % ('Epoch', 'ratio', 'iter', 'alliter', 'lr', 'Class_L', 'Confi_L', "iouL", "Loss", "avgl", \
                "iounow", "Confi", "NO_Confi", "Class"))
        # if rank == 0:
        #     flogs.write('Epoch {}/{}'.format(epoch, num_epochs)+'\n')
        #     print('Epoch {}/{}'.format(epoch, num_epochs))
        #     print('-'*10)
        running_loss = 0
        if epoch<=nowepoch:
            stepiters += len(dataloader)
            continue
        count = 0
        model.train()
        losscol = []
        optimizer.zero_grad()

        dataloader.sampler.set_epoch(epoch)
        lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, batch_size, momnetum, learning_rate)
        # if epoch == warmepoch + 1:
        #     optimizer.momentum = momnetum
        # preiou = 0
        pbar = enumerate(dataloader)
        if rank==0:
            TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
            pbar = tqdm.tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)
        # for i, (image, labels, _) in enumerate(dataloader):
        for i, (image, labels, _) in pbar:
        # for i, (image, labels, _) in enumerate(tqdm.tqdm(dataloader, desc=f"{epoch}/{num_epochs}", mininterval=3)):
            stepiters += 1
            # if stepiters < alliters:
            #     continue
            # cvshow_(image, labels)   #cv2 show inputs images)
            count += 1
            if epoch <= warmepoch:
                lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, batch_size, momnetum, learning_rate)
            
            image = image.to(devicenow)
            labels = labels.to(devicenow)
            
            # try:
            with torch.cuda.amp.autocast(amp):
                prediction = model(image, yolovfive = yolovfive)

                # except Exception as e:
                #     continue
                try:
                    # loss, c_l, confi_l, iouloss = calculate_losses_yolov3(prediction, labels, model, count_scale)
                    if chooseLoss == "darknetRevise":
                        loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_darknetRevise(prediction, labels, model, ignore_thresh, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                    elif chooseLoss == "darknet":
                        loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_darknet(prediction, labels, model, ignore_thresh, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                    elif chooseLoss == "Alexeydarknet":
                        loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_Alexeydarknet(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                    elif chooseLoss == "yolofive":
                        loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_yolofive(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                                                                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                        # loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_yolofive_revise(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                        #                                                                     bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                    elif chooseLoss == "20230730":
                        loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_20230730(prediction, labels, model, count_scale, ignore_thresh, \
                                                                                        bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                    else:
                        exit(-1)

                    # loss, loss_components = computeloss(prediction, labels, devicenow, model)
                except Exception as e:
                    print(e)
                    savestate = {'state_dict':model.state_dict(),\
                                        'iteration':i,\
                                        'alliters':stepiters,\
                                        'nowepoch':epoch}
                    __savepath__ = os.path.join(savepath, datekkk) + prefix
                    os.makedirs(__savepath__, exist_ok=True)
                    torch.save(savestate, __savepath__+os.sep+r'model_e{}_l{:.3f}.pt'.format(epoch, epoch_loss))
                    continue
            
            # if darknetLoss and i > 30 and iouloss > 10 and iouloss / preiou > 2:
            #     loss = c_l + confi_l
            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if rank != -1:
                # print(WORLD_SIZE)
                loss *= cudanum  # 6 # gradient averaged between devices in DDP mode

            if amp:
                scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
            if amp:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
            else:
                optimizer.step() #C:\Users\10696\Desktop\Pytorch_YOLOV3\\datas\train\images\2010_003635.jpg                
            optimizer.zero_grad()

            if ema and rank==0:
                ema.update(model)
            # preiou = iouloss

            losscol.append(loss.detach().cpu().item())
            loss = loss.detach().cpu().item()
            running_loss += loss
            epoch_loss = running_loss / count
#             logword = '''\nepoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.3f}, MSE loss: {:.3f}, Class loss: {:.3f}, \
# Confi loss: {:.3f}, iouloss: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}'''.format(
#                                    epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], float(mse.item()),\
#                                     float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss)
            logword = '''\ne: {}, r:{:.2f}%, i: {}, ai: {}, lr: {:.6f}, Class: {:.3f}, \
Confi: {:.3f}, iou: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}, iounow: {:.3f}, cof: {:.3f}, ncof: {:.6f}, cla: {:.3f}'''.format(
                                   epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], \
                                    float(c_l.item()/boxnum), float(confi_l.item()/boxnum), iouloss.item()/boxnum, loss/boxnum, \
                                        epoch_loss/boxnum, iounow.item(), cof.item(), ncof.item(), cla.item())
            # if i%subsiz==0 or i == len(dataloader) - 1:
            if rank == 0 and i%1==0:
                # print(logword)
                pbar.set_description(('%9s'+'%9.5g'+"%9s"+"%9s"+ '%9.5g' * 10) %
                                     (epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], \
                                    float(c_l.item()/boxnum), float(confi_l.item()/boxnum), iouloss.item()/boxnum, loss/boxnum, \
                                        epoch_loss/boxnum, iounow.item(), cof.item(), ncof.item(), cla.item()))
                flogs.write(logword+'\n')
        # if rank==0 and ema:
        #     ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

        if rank == 0 and epoch==num_epochs:
            savestate = {'state_dict':deepcopy(de_parallel(model)).half(),\
                            'ema':deepcopy(ema.ema).half() if ema else "", \
                            'alliters':stepiters,\
                            'nowepoch':epoch}
            __savepath__ = os.path.join(savepath, datekkk) + prefix
            os.makedirs(__savepath__, exist_ok=True)
            # torch.save(savestate, __savepath__+os.sep+r'model_e{}_l{:.3f}.pt'.format(epoch, epoch_loss))
        if rank == 0:
            jkk = []
            if os.path.exists(os.path.join(abspath, 'valid.txt')):
                with open(os.path.join(abspath, 'valid.txt'), 'r') as obj:
                    kk = obj.read()
                    kk = kk.strip().split(',')
                    while "" in kk:
                        kk.remove("")
                    for ki in kk:
                        try:
                            jkk.append(int(ki))
                        except:
                            pass
        if rank == 0 and (epoch in jkk or epoch==num_epochs):
            savestate = {'state_dict':deepcopy(de_parallel(model)).half(),\
                            'ema':deepcopy(ema.ema).half() if ema else "", \
                            'alliters':stepiters,\
                            'nowepoch':epoch}
            __savepath__ = os.path.join(savepath, datekkk) + prefix
            os.makedirs(__savepath__, exist_ok=True)
            # scheduler.step(np.mean(losscol))
            [map, mAP0], lengthkk = validation_map(model if ema==None else ema.ema, yolovfive, valdataloader, devicenow) #, score_thresh, nms_thresh)
            # map = evaluation(model, score_thresh_now = 0.01, nms_thresh_now = 0.3)
            print("validation......num_img: {}, mAP: {}, premap:{}".format(lengthkk, [map, mAP0], pre_map))
            if len(map) > 2:
                pcmap = [round(map[0], 6), round(np.mean(map), 6)]
            strmap = str(pcmap).replace(",", "_").replace(" ", "_")

            # torch.save(savestate, __savepath__+os.sep+r'model_e{}_map{}_l{:.3f}_{}.pt'.format(epoch, strmap, epoch_loss, datekkk))
            # if(pre_map < np.mean(map)) or (epoch+1)%1==0 or epoch==num_epochs:
            torch.save(savestate, __savepath__+os.sep+r'model_e{}_map{}_l{:.3f}_{}.pt'.format(epoch, strmap, epoch_loss, datekkk))
            print('savemodel ')
            pre_map = np.mean(map)
            del savestate

    dist.destroy_process_group()
    if rank == 0:
        timeused  = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
        flogs.close()

if __name__ == '__main__':
    # mp.spawn(demo_fn,
    #          args=(world_size,),
    #          nprocs=world_size,
    #          join=True)
    # port = find_free_port()
    # if len(rank) > 1:
    #     port = find_free_port()
    #     mp.spawn(trainer, args=([port]), nprocs=2,)
    # else:
    trainer()