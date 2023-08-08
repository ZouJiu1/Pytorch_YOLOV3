#encoding=utf-8
#Author：ZouJiu
#Time: 2023-7-27

import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

from config.config_yolov3_20230727 import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import time
import torch
import numpy as np
from models.Yolov3_20230727 import Yolov3Net
from models.layer_loss import calculate_losses_yolov3, calculate_losses_darknet, calculate_losses_Alexeydarknet, calculate_losses_20230730, calculate_losses_yolofiveeight
import torch.optim as optim
from utils.common import cvshow_, collate_fn, provide_determinism
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

def adjust_lr(optimizer, stepiters, epoch, num_batch, num_epochs, Adam, freeze_backbone, \
              momnetum, learning_rate, model, weight_decay):
    steps0 = num_batch * warmepoch - 1
    final_lr = 0.1
    baselr = learning_rate
    if epoch <= warmepoch:
        # lr = baselr * np.exp(stepiters * 6 / steps0 - 6)
        lr = (baselr / steps0) * stepiters
    # else:
        # lr = ((1 - math.cos(epoch * math.pi / num_epochs)) / 2) * (final_lr - 1) + 1
        # lr = baselr * lr
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
    python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 --master_addr="172.17.0.2" --master_port=55568 allcodes/train_yolov3_20230727_distribute.py
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="172.17.0.2" --master_port=55568 allcodes/train_yolov3_20230727_distribute.py
    '''
    # https://pytorch.org/docs/stable/distributed.html#launch-utility
    # https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    os.system("export NCCL_SOCKET_IFNAME=eth0")
    os.system("export TORCH_DISTRIBUTED_DEBUG=DETAIL") #DEBUG
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    cudanum = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    assert cudanum == torch.cuda.device_count()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    # exit(0)
    if seed != -1:
        provide_determinism(seed)
    torch.cuda.manual_seed_all(999999999)
    torch.manual_seed(999999999)
    
    torch.cuda.set_device(rank % torch.cuda.device_count())
    devicenow = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")

    #pip3 install --user --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    traindata = trainDataset(trainpath, train_imgpath, stride = strides, anchors = anchors, \
                            augment = False, inputwidth = inputwidth, transform=TF)
    num_cpu = 16 #num_cpu if num_cpu < 20 else 13
    count_scale = traindata.count_scale.to(devicenow)
    
    train_sampler = DistributedSampler(dataset=traindata, shuffle=True)
    dataloader = DataLoader(traindata, batch_size = batch_size // subsiz, shuffle=False, \
        num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)

    if rank==0:
        valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                    augment = False, inputwidth = inputwidth, transform=TFRESIZE)
        # val_sampler = DistributedSampler(dataset=valdata, shuffle=True)
        valdataloader = DataLoader(valdata, batch_size = batch_size // subsiz, shuffle=False, \
                    num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True) #, sampler=val_sampler)

    if rank == 0:
        flogs = open(logfile, 'w')
    model = Yolov3Net(num_classes, anchors, devicenow, inputwidth)
    # yolov3 = Yolov3().to(self.devicenow)
    print(model)
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

    # Convert BatchNorm to SyncBatchNorm. 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(devicenow)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank) #, find_unused_parameters=True)

    # params = model.parameters() # [p for p in model.parameters() if p.requires_grad]
    # if Adam:
    #     optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    # else:
    
    # pg0, pg1, pg2 = [], [], []
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)

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
    print('Using {} device'.format(devicenow))
    length = len(dataloader)
    if rank == 0:
        flogs.write('Using {} device'.format(devicenow)+'\n')
    stepiters = 0
    pre_map = 0

    bce0loss = torch.nn.BCELoss(reduction='sum').to(devicenow)
    bce1loss = torch.nn.BCELoss(reduction='sum').to(devicenow)
    bce2loss = torch.nn.BCELoss(reduction='sum').to(devicenow)
    bcecls = torch.nn.BCELoss(reduction='sum').to(devicenow)
    bcecof = torch.nn.BCELoss(reduction='sum').to(devicenow)
    mseloss = [torch.nn.MSELoss(reduction='sum').to(device) for i in range(2*2)]
    epoch_loss = 0
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        if rank == 0:
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

        dataloader.sampler.set_epoch(epoch)
        lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay)
        if epoch <= warmepoch:
            optimizer.momentum = warmup_momnetum
        elif epoch == warmepoch + 1:
            optimizer.momentum = momnetum
        # for i, (image, labels) in enumerate(dataloader):
        for i, (image, labels, _) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):
            stepiters += 1
            # if stepiters < alliters:
            #     continue
            # cvshow_(image, labels)   #cv2 show inputs images)
            count += 1
            if epoch <= warmepoch:
                lr = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay)
            
            image = image.to(devicenow)
            labels = labels.to(devicenow)
            
            # try:
            prediction = model(image)
            # except Exception as e:
            #     continue
            
            try:
                # loss, c_l, confi_l, iouloss = calculate_losses_yolov3(prediction, labels, model, count_scale)
                loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_darknet(prediction, labels, model, ignore_thresh, \
                                                                                    bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                # loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_Alexeydarknet(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                #                                                                     bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                # loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_yolofiveeight(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                #                                                                     bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                # loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, boxnum = calculate_losses_20230730(prediction, labels, model, count_scale, ignore_thresh, \
                #                                                                     bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss)
                
                # loss, loss_components = computeloss(prediction, labels, devicenow, model)
            except Exception as e:
                print(e)
                savestate = {'state_dict':model.state_dict(),\
                                    'iteration':i,\
                                    'alliters':stepiters,\
                                    'nowepoch':epoch}
                __savepath__ = os.path.join(savepath, datekkk) + prefix
                os.makedirs(__savepath__, exist_ok=True)
                torch.save(savestate, __savepath__+os.sep+r'model_e{}_l{:.3f}.pth'.format(epoch, epoch_loss))
                continue

            losscol.append(loss.detach().cpu().item())
            
            loss.backward()
            loss = loss.detach().cpu().item()
            running_loss += loss
            epoch_loss = running_loss / count
#             logword = '''\nepoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.6f}, MSE loss: {:.6f}, Class loss: {:.3f}, \
# Confi loss: {:.3f}, iouloss: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}'''.format(
#                                    epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], float(mse.item()),\
#                                     float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss)
            logword = '''\ne: {}, r:{:.2f}%, i: {}, ai: {}, lr: {:.6f}, Class: {:.3f}, \
Confi: {:.3f}, iou: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}, iounow: {:.3f}, cof: {:.3f}, ncof: {:.6f}, cla: {:.3f}'''.format(
                                   epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], \
                                    float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss, iounow.item(), cof.item(), ncof.item(), cla.item())
            # if i%subsiz==0 or i == len(dataloader) - 1:
            optimizer.step() #C:\Users\10696\Desktop\Pytorch_YOLOV3\\datas\train\images\2010_003635.jpg
            optimizer.zero_grad()
            if rank == 0 and i%3==0:
                print(logword)
                flogs.write(logword+'\n')
                flogs.flush()

        if rank == 0:
            savestate = {'state_dict':model.state_dict(),\
                            'iteration':i,\
                            'alliters':stepiters,\
                            'nowepoch':epoch}
            __savepath__ = os.path.join(savepath, datekkk) + prefix
            os.makedirs(__savepath__, exist_ok=True)
            torch.save(savestate, __savepath__+os.sep+r'model_e{}_l{:.3f}.pth'.format(epoch, epoch_loss))
        if rank == 0:
            jkk = []
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
            savestate = {'state_dict':model.state_dict(),\
                            'iteration':i,\
                            'alliters':stepiters,\
                            'nowepoch':epoch}
            __savepath__ = os.path.join(savepath, datekkk) + prefix
            os.makedirs(__savepath__, exist_ok=True)
            # scheduler.step(np.mean(losscol))
            map, lengthkk = validation_map(model, valdataloader, devicenow) #, score_thresh, nms_thresh)
            # map = evaluation(model, score_thresh_now = 0.01, nms_thresh_now = 0.3)
            print("validation......num_img: {}, mAP: {}, premap:{}".format(lengthkk, map, pre_map))
            if len(map) > 2:
                pcmap = [round(map[0], 6), round(np.mean(map), 6)]
            strmap = str(pcmap).replace(",", "_").replace(" ", "_")

            torch.save(savestate, __savepath__+os.sep+r'model_e{}_map{}_l{:.3f}_{}.pth'.format(epoch, strmap, epoch_loss, datekkk))
            if(pre_map < np.mean(map)) or (epoch+1)%1==0 or epoch==num_epochs:
                torch.save(savestate, __savepath__+os.sep+r'model_e{}_map{}_l{:.3f}_{}.pth'.format(epoch, strmap, epoch_loss, datekkk))
                print('savemodel ')
                pre_map = np.mean(map)
            del savestate
        # except:
        #     print('error: don\'t savemodel')
        # evaluate(model, dataloader_test, devicenow = devicenow)

    dist.destroy_process_group()
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    if rank == 0:
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
