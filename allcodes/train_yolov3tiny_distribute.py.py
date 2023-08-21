#encoding=utf-8
#Author：ZouJiu
#Time: 2022-12-10

import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

from config.config_yolov3tiny import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

import time
import torch
import numpy as np
from models.Yolov3tiny import yolov3tinyNet
from models.layer_loss_20230816 import calculate_losses_yolov3, calculate_losses_darknet, calculate_losses_Alexeydarknet
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
              momnetum, learning_rate, model, weight_decay, flogs):
    steps0 = num_batch * warmepoch - 1
    final_lr = 0.01
    baselr = learning_rate
    if epoch <= warmepoch:
        lr = baselr * np.exp(stepiters * 6 / steps0 - 6)
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

    return optimizer

def trainer():
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_DEBUG=INFO
    NCCL_SOCKET_IFNAME=eth1  # 网卡名称更换您自己的
    NCCL_DEBUG=WARN
    python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="172.17.0.10" --master_port=55568 allcodes/train_yolov3tiny_distribute.py
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="172.17.0.10" --master_port=55568 allcodes/train_yolov3tiny_distribute.py
    '''
    # https://pytorch.org/docs/stable/distributed.html#launch-utility

    # os.system("export NCCL_SOCKET_IFNAME=eth0")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")
    # exit(0)
    if seed != -1:
        provide_determinism(seed)
    # torch.cuda.manual_seed_all(999999999)
    # torch.manual_seed(999999999)
    devicenow = rank % torch.cuda.device_count()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    devicenow = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    # synchronizes all the threads to reach this point before moving on
    # dist.barrier()

    #pip3 install --user --upgrade opencv-python -i https://pypi.tuna.tsinghua.edu.cn/simple
    traindata = trainDataset(trainpath, train_imgpath, stride = strides, anchors = anchors, \
                            augment = False, inputwidth = inputwidth, transform=TF)
    num_cpu = cpu_count()
    num_cpu =  26 -2 #num_cpu if num_cpu < 20 else 13
    # devicenow = "cuda" #if torch.cuda.is_available() else "cpu"
    count_scale = traindata.count_scale.to(devicenow)
    
    train_sampler = DistributedSampler(dataset=traindata, shuffle=True)
    dataloader = DataLoader(traindata, batch_size = batch_size,shuffle=False, \
        num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True, sampler=train_sampler)

    valdata = trainDataset(pth_evaluate, img_evaluate, stride = strides, anchors = anchors, \
                                augment = False, inputwidth = inputwidth, transform=TFRESIZE)
    val_sampler = DistributedSampler(dataset=valdata, shuffle=True)
    valdataloader = DataLoader(valdata, batch_size = batch_size, shuffle=False, \
                num_workers=num_cpu, collate_fn=collate_fn, pin_memory=True, sampler=val_sampler)

    flogs = open(logfile, 'w')
    model = yolov3tinyNet(num_classes, anchors, devicenow, inputwidth)
    # yolov3 = Yolov3().to(self.devicenow)
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
        pretrained = state_dict['state_dict']
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

    model = model.to(devicenow)

    # Convert BatchNorm to SyncBatchNorm. 
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    local_rank = [int(i) for i in os.environ['LOCAL_RANK'].split(",")]
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    params = [p for p in model.parameters() if p.requires_grad]
    if Adam:
        optimizer = optim.Adam(params, lr=learning_rate, betas=(momnetum, 0.999), weight_decay= weight_decay)  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(params, lr=learning_rate, momentum=momnetum, nesterov=True, weight_decay= weight_decay)
        
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
    flogs.write('Using {} device'.format(devicenow)+'\n')
    stepiters = 0
    pre_map = 0
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
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
        
        dataloader.sampler.set_epoch(epoch)
        optimizer = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay, flogs)
        # for i, (image, labels) in enumerate(dataloader):
        for i, (image, labels, _) in enumerate(tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}/{num_epochs}")):
            stepiters += 1
            # if stepiters < alliters:
            #     continue
            # cvshow_(image, labels)   #cv2 show inputs images)
            count += 1
            if epoch <= warmepoch:
                optimizer = adjust_lr(optimizer, stepiters, epoch, len(dataloader), num_epochs, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay, flogs)
            
            image = image.to(devicenow)
            labels = labels.to(devicenow)
            
            # try:
            prediction = model(image)
            # except Exception as e:
            #     continue
            
            # loss, mse, c_l, confi_l, iouloss = calculate_losses_yolov3(prediction, labels, model, count_scale)
            # loss, mse, c_l, confi_l, iouloss = calculate_losses_darknet(prediction, labels, model, ignore_thresh)
            loss, mse, c_l, confi_l, iouloss = calculate_losses_Alexeydarknet(prediction, labels, model, ignore_thresh, iou_thresh, count_scale)
            # loss, mse, c_l, confi_l, iouloss = calculate_losses(prediction, labels, model, count_scale)
            
            # loss, loss_components = computeloss(prediction, labels, devicenow, model)
            losscol.append(loss.detach().cpu().item())
            
            # if len(devicenow) > 1:
            #     loss *= WORLD_SIZE
            loss.backward()
            # loss.requires_grad_(True)
            # loss = loss.to(devicenow)
            # if torch.isnan(loss).item()==False:
            loss = loss.detach().cpu().item()
            running_loss += loss
            # else:
            #     optimizer = adjust_lr(optimizer, 200, epoch, Adam, freeze_backbone, momnetum, learning_rate, model, weight_decay, flogs)
            #     print(torch.isnan(loss).item(), torch.isnan(loss).item()==False)
            # statistics
            epoch_loss = running_loss / count
#             logword = '''\nepoch: {}, ratio:{:.2f}%, iteration: {}, alliters: {}, lr: {:.6f}, MSE loss: {:.6f}, Class loss: {:.3f}, \
# Confi loss: {:.3f}, iouloss: {:.3f}, Loss: {:.3f}, avgloss: {:.3f}'''.format(
#                                    epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], float(mse.item()),\
#                                     float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss)
            logword = '''\ne: {}, r:{:.2f}%, i: {}, ai: {}, lr: {:.6f}, MSE: {:.1f}, Class: {:.6f}, \
Confi: {:.6f}, iou: {:.6f}, Loss: {:.6f}, avgloss: {:.6f}'''.format(
                                   epoch, i*100/length, i+1, stepiters, optimizer.state_dict()['param_groups'][0]['lr'], float(mse.item()),\
                                    float(c_l.item()), float(confi_l.item()), iouloss.item(), loss, epoch_loss)
            if i%subsiz==0 or i == len(dataloader) - 1:
                optimizer.step() #C:\Users\10696\Desktop\Pytorch_YOLOV3\\datas\train\images\2010_003635.jpg
                optimizer.zero_grad()
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
            # scheduler.step(np.mean(losscol))
            map, lengthkk = validation_map(model, valdataloader, devicenow, score_thresh, nms_thresh)
            # map = evaluation(model, score_thresh_now = 0.01, nms_thresh_now = 0.3)
            print("validation......num_img: {}, mAP: {}, premap:{}".format(lengthkk, map, pre_map))
            if len(map) > 2:
                map = [round(map[0], 6), round(np.mean(map), 6)]
            strmap = str(map).replace(",", "_").replace(" ", "_")

            torch.save(savestate, __savepath__+os.sep+r'model_e{}_t{}_map{}_l{:.3f}_{}.pth'.format(epoch, stepiters, strmap, epoch_loss, datekkk))
            if(pre_map < np.mean(map)) or (epoch+1)%1==0 or epoch==num_epochs-1:
                torch.save(savestate, __savepath__+os.sep+r'model_e{}_t{}_map{}_l{:.3f}_{}.pth'.format(epoch, stepiters, strmap, epoch_loss, datekkk))
                print('savemodel ')
                pre_map = np.mean(map)
            del savestate
        # except:
        #     print('error: don\'t savemodel')
        # evaluate(model, dataloader_test, devicenow = devicenow)

    dist.destroy_process_group()
    timeused  = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(timeused//60, timeused%60))
    flogs.close()

def tmp():
    pass

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
