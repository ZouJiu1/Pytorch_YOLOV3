import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import cv2
import torch
import torch.nn as nn
import numpy as np
try:
    from torchvision.ops.boxes import complete_box_iou, box_iou
except:
    from models.layer_yolo import complete_box_iou, box_iou
from typing import Tuple
from torch import Tensor
from torch.multiprocessing import Pool, set_start_method
import torch.nn.functional as F

def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def _box_area_(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union_(boxes1: torch.Tensor, boxes2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    area1 = _box_area_(boxes1)
    area2 = _box_area_(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    return inter, union

def _box_noexpand_iou_(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[N, 4]): second set of boxes

    Returns:
        Tensor[N]: the N the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union_(boxes1, boxes2)
    iou = inter / union
    return iou, union

def _box_noexpand_diou_iou_(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:

    iou, union = _box_noexpand_iou_(boxes1, boxes2)
    lti = torch.min(boxes1[:, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N,2]
    c_giou = whi[:, 0] * whi[:, 1]
    diagonal_distance_squared = (whi[:, 0] ** 2) + (whi[:, 1] ** 2) + eps
    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p - x_g)) ** 2) + (
        _upcast((y_p - y_g)) ** 2
    )
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    
    giou_term = (c_giou - union) / c_giou
    giou = iou - giou_term
    return iou - (centers_distance_squared / diagonal_distance_squared), iou, giou

def complete_box_iou_no_expand(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[N, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N]: the N array containing the complete IoU values
        for every element in boxes1 and boxes2
    """

    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou, giou = _box_noexpand_diou_iou_(boxes1, boxes2, eps)

    w_pred = boxes1[:, 2] - boxes1[:, 0]
    h_pred = boxes1[:, 3] - boxes1[:, 1]

    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / (h_pred + 1e-10)) - torch.atan(w_gt / (h_gt + 1e-10)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v, diou, iou, giou

def xywh2xyxy(inputs, imgsize, clamp=True):
    min_xy = inputs[:, 0:2] - inputs[:, 2:] / 2.0
    max_xy = inputs[:, 0:2] + inputs[:, 2:] / 2.0
    inputs[:, 0:2] = min_xy
    inputs[:, 2:]  = max_xy
    # if clamp:
    #     inputs = torch.clamp(inputs, 0, imgsize-1)
    return inputs

def calculate_losses_largearea(prediction, labels, model, count_scale, \
                                bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    predicts = []
    anchors = []
    for i in range(len(model.yolo)):
        anchors.append(prediction[i][1])
        predicts.append(prediction[i][0])
    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    batchsize = prediction[0][0].size()[0]
    
    mseloss = torch.nn.MSELoss(reduction='none').to(model.device)
    bceloss = torch.nn.BCELoss().to(model.device)
    bcecls = torch.nn.BCELoss(reduction='none').to(model.device)
    bcecof = torch.nn.BCELoss(reduction='none').to(model.device)

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    thresh = 0.6 - 0.2 - 0.1
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    n, k, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    all_iou = complete_box_iou(anchors, labels[:, 2:])
    pre = 0
    maxind = torch.tensor([], dtype=torch.int).to(model.device)
    col_choose = torch.tensor([], dtype=torch.bool).to(model.device)
    iou_scale = torch.tensor([], dtype=torch.float).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        iou = all_iou[:, pre:pre + num]
        
        max_val, max_ind  = torch.max(iou, dim = 1)
        choose = max_val > thresh

##############-------------------------------------###############################################
        T_iou = iou.T
        ########################
        # T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
        # max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
        # choose[...] = False
        # choose[T_max_ind] = True
        ########################
        # dic = {}
        # for ij in max_ind[choose]:
        #     ij = ij.cpu().item()
        #     if ij not in dic.keys():
        #         dic[ij] = [1, 0]
        #     else:
        #         dic[ij][0] += 1 
        arg = torch.argsort(T_iou, dim = 1, descending=True)[:, :3]
        for ia in range(len(arg)):
            max_ind[arg[ia]] = ia
            choose[arg[ia]] = True
        # for ij in max_ind[choose]:
        #     ij = ij.cpu().item()
        #     if ij not in dic.keys():
        #         dic[ij] = [0, 1]
        #     else:
        #         dic[ij][1] += 1
##############-------------------------------------###############################################
        
        while len(max_ind[choose]) == 0:
            thresh = np.exp(-thresh) * thresh
            max_val, max_ind  = torch.max(iou, dim = 1)
            choose = max_val > thresh
            T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
            max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
            choose[T_max_ind] = True
        
        maxind = torch.concat([maxind, max_ind[choose] + pre], dim = 0)
        col_choose = torch.concat([col_choose, choose], dim = 0)
        iou_column = iou[choose]
        iou_ch = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        iou_scale = torch.concat([iou_scale, iou_ch], dim = 0)
        
        pre += num

    choose_predict = predicts[col_choose, :]
    choose_label   = labels[maxind, :]
    
    count_scale = count_scale[choose_label[:, 1].long()] # / 10.0

##########################
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]

    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    for i in torch.unique(maxind):
        ch = maxind==i
        iounow = iou_scale[ch]
        iouch = iounow / torch.max(iounow)
        # sum = int(torch.sum(ch))
        # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
        iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
##########################

    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)
    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    prediou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))
    iou_loss = (1 - prediou) * iou_scale         # scale * count_scale
    iouloss += torch.mean(iou_loss)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2*2+1):]  # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    kk = predicts[~col_choose, (2*2+1):]
    iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # count_scale = torch.unsqueeze(count_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) + bceloss(kk, torch.zeros_like(kk))

    confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    noconf = predicts[~col_choose, (2*2)].unsqueeze(-1)
    confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * iou_scale) + bceloss(noconf, torch.zeros_like(noconf))
    # prediou = prediou.unsqueeze(-1)
    # cofobj = torch.ones_like(confidence) * prediou.clamp(0).type(confidence.dtype)  * count_scale
    # confi_l   += torch.mean(bcecof(confidence, cofobj)) + bceloss(noconf, torch.zeros_like(noconf))

    if batchsize:
        mse /= batchsize
        c_l /= batchsize
        confi_l /= batchsize
        iouloss /= batchsize

    confi_l *= 1.0
    c_l *= 0.6
    iouloss *= 0.06
    loss = ( c_l + confi_l + iouloss ) * batchsize
    return loss, mse, c_l, confi_l, iouloss

def calculate_losses_yolov3(prediction, labels, model, count_scale):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    predicts = []
    anchors = []
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    
    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]

    cx = []
    cy = []
    cax = [] 
    cay = []
    batchsize = prediction[0][0].size()[0]
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        predicts.append(prediction[i][0])

        center_truth_x = (((labels[:, 2] / strides[i]) + (labels[:, 2*2] / strides[i])) / 2).long()
        center_truth_x = center_truth_x.T.unsqueeze(-1)
        center_truth_x = torch.repeat_interleave(center_truth_x, repeats = len(anc), dim = -1)
        cx.append(center_truth_x)
    
        center_truth_y = (((labels[:, 3] / strides[i]) + (labels[:, 2*2+1] / strides[i])) / 2).long()
        center_truth_y = center_truth_y.T.unsqueeze(-1)
        center_truth_y = torch.repeat_interleave(center_truth_y, repeats = len(anc), dim = -1)
        cy.append(center_truth_y)

        center_anchor_x = ((anc[:, 0] / strides[i] + anc[:, 2] / strides[i]) / 2).unsqueeze(0).long()
        center_anchor_x = torch.repeat_interleave(center_anchor_x, repeats=len(labels), dim=0)
        cax.append(center_anchor_x)
        
        center_anchor_y = ((anc[:, 1] / strides[i] + anc[:, 3] / strides[i]) / 2).unsqueeze(0).long()
        center_anchor_y = torch.repeat_interleave(center_anchor_y, repeats=len(labels), dim=0)
        cay.append(center_anchor_y)

    cx = torch.concat(cx, dim=(-1))
    cy = torch.concat(cy, dim=(-1))
    cax = torch.concat(cax, dim=(-1))
    cay = torch.concat(cay, dim=(-1))
    xmask = cx == cax
    # k = torch.sum(xmask)
    ymask = cy == cay
    # kk = torch.sum(ymask)
    mask = xmask & ymask
    # kkmask = torch.sum(mask)
    del xmask, ymask, cx, cy, cax, cay, prediction

    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    
    # mseloss = torch.nn.MSELoss(reduction='none').to(model.device)
    bceloss = torch.nn.BCELoss().to(model.device)
    # bcecls = torch.nn.BCELoss(reduction='none').to(model.device)
    # bcecof = torch.nn.BCELoss(reduction='none').to(model.device)

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    # thresh = 0.6 - 0.2 - 0.1
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    n, k, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    all_iou = complete_box_iou(labels[:, 2:], anchors)
    # all_iou = box_iou(anchors, labels[:, 2:])
    all_iou = all_iou * mask
    pre = 0
    maxind = torch.tensor([], dtype=torch.int).to(model.device)
    col_choose = torch.tensor([], dtype=torch.bool).to(model.device)
    iou_scale = torch.tensor([], dtype=torch.float).to(model.device)
    
    Tvalue_all, Tindexs_all = torch.sort(all_iou, dim = 1, descending=True)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        chnum = len(model.yolo)
        choose = torch.zeros(len(anchors), dtype = torch.bool, device = model.device)
        max_ind = torch.zeros(len(anchors), dtype = torch.int, device = model.device)
        # Tvalue = Tvalue_all[pre: pre+num, :]
        Tindexs = Tindexs_all[pre: pre+num, :]
        arg = Tindexs[:, :chnum]
        # argkkk = Tvalue[:, :chnum]
        for ia in range(len(arg)):
            argind = arg[ia] #[argkkk[ia] > 0]
            max_ind[argind] = ia
            choose[argind] = True
##############-------------------------------------###############################################

        # index = labels[:, 0].long() == i
        # num = torch.sum(index)
        # la = labels[index].clone()
        # xyxy = labels[index, 2:].clone()
        # tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * (260 - 2*2 - 1)
        # import shutil
        # try:
        #     shutil.rmtree(r'/root/project/Pytorch_YOLOV3/datas/imshow')
        # except:
        #     pass
        # os.makedirs(r'/root/project/Pytorch_YOLOV3/datas/imshow', exist_ok=True)
        # # for k in range(len(anchors)):
        # #     if (k+1)%10==0:
        # #         cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/%s.jpg'%str(k//10), tmp)
        # #         tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * 2**(2**3)
        # #     cv2.rectangle(tmp, (int(anchors[k][0]), int(anchors[k][1])), \
        # #         (int(anchors[k][2]), int(anchors[k][3])), \
        # #         [np.random.randint(255),np.random.randint(255),np.random.randint(255)], 2)
        #     # if k == 100:
        #     #     break
        # tmp = np.ones((model.imgsize, model.imgsize), dtype=np.uint8) * (260 - 2*2 - 1)
        # for k in range(len(xyxy)):
        #     cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
        #         (int(xyxy[k][2]), int(xyxy[k][3])), (0,0,255), 2)
        # cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth.jpg', tmp)
        
        # anchors = anchors[choose]
        # xyxy    = xyxy[max_ind[choose]]
        # cvfont = cv2.FONT_HERSHEY_SIMPLEX
        # for k in range(len(anchors)):
        #     if k == len(anchors) - 1:
        #         k = k
        #     tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * (260 - 2*2 - 1)
        #     cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
        #         (int(xyxy[k][2]), int(xyxy[k][3])), (255,126,255), 2)
        #     cv2.rectangle(tmp, (int(anchors[k][0]), int(anchors[k][1])), \
        #         (int(anchors[k][2]), int(anchors[k][3])), (255,0,0), 1)
        #     cx = ((xyxy[k][0] + xyxy[k][2]) / 2 / 16).long()
        #     cx3 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 32).long()
        #     cy = ((xyxy[k][1] + xyxy[k][3]) / 2 / 16).long()
        #     cy3 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 32).long()
        #     kk = (cx, cy)
        #     kk3 = (cx3, cy3)
            
        #     acx = ((anchors[k][0] + anchors[k][2]) / 2 / 16).long()
        #     acx3 = ((anchors[k][0] + xyxy[k][2]) / 2 / 32).long()
        #     acy = ((anchors[k][1] + anchors[k][3]) / 2 / 16).long()
        #     acy3 = ((anchors[k][1] + anchors[k][3]) / 2 / 32).long()
        #     akk = (acx, acy)
        #     akk3 = (acx3, acy3)
        #     cv2.putText(tmp, str(max_ind[choose][k]), (int(xyxy[k][0]), int(xyxy[k][1]) + 10), cvfont, 0.5, [255, 0, 0], 1)
        #     cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth_%d.jpg'%k, tmp)
        # exit(0)
        
        maxind = torch.concat([maxind, max_ind[choose] + pre], dim = 0)
        col_choose = torch.concat([col_choose, choose], dim = 0)
        # iou_column = iou[choose]
        # iou_ch = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        # iou_scale = torch.concat([iou_scale, iou_ch], dim = 0)
        
        pre += num

    col_choose_rev = ~col_choose
    choose_predict = predicts[col_choose, :]
    choose_label   = labels[maxind.long(), :]
    
    del mask, all_iou, Tindexs_all, Tvalue_all, col_choose, maxind
    
    # count_scale = count_scale[choose_label[:, 1].long()] # / 10.0

##########################
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]

    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    # for i in torch.unique(maxind):
    #     ch = maxind==i
    #     iounow = iou_scale[ch]
    #     iouch = iounow / torch.max(iounow)
    #     # sum = int(torch.sum(ch))
    #     # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
    #     iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
##########################

    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)
    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    prediou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))
    iou_loss = (1 - prediou)  # * iou_scale         # scale * count_scale
    iouloss += torch.mean(iou_loss)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2*2+1):]  # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    kk = predicts[col_choose_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # count_scale = torch.unsqueeze(count_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    c_l   += bceloss(classes, class_la) + bceloss(kk, torch.zeros_like(kk))

    confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    noconf = predicts[col_choose_rev, (2*2)].unsqueeze(-1)
    confi_l   += bceloss(confidence, torch.ones_like(confidence)) + bceloss(noconf, torch.zeros_like(noconf))
    # prediou = prediou.unsqueeze(-1)
    # cofobj = torch.ones_like(confidence) * prediou.clamp(0).type(confidence.dtype)  * count_scale
    # confi_l   += torch.mean(bcecof(confidence, cofobj)) + bceloss(noconf, torch.zeros_like(noconf))

    if batchsize:
        mse /= batchsize
        c_l /= batchsize
        confi_l /= batchsize
        iouloss /= batchsize

    confi_l *= 1.0
    c_l *= 0.6
    iouloss *= 0.06
    loss = ( c_l + confi_l + iouloss ) * batchsize
    return loss, mse, c_l, confi_l, iouloss

def cal_ignore_pre(truth, predict, ignore_thresh):
    kk = complete_box_iou(truth, predict) <= ignore_thresh
    return kk

def calculate_losses_darknetRevise(prediction, labels, model, ignore_thresh, \
                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    predicts = []
    anchors = []
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    labels = torch.concat([labels, torch.arange(len(labels), device = model.device).unsqueeze(-1)], dim = -1)

    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]
    mapsize = [(model.imgsize / i) for i in strides]
    anchor_per_layer = 3
    batchsize = prediction[0][0].size()[0]
    truthind = torch.tensor([], dtype=torch.long).to(model.device)
    anchorsind = torch.tensor([], dtype=torch.long).to(model.device)
    predictind = torch.tensor([], dtype=torch.long).to(model.device)
    
    pre_num = 0
    total_pixel = 0
    for i in range(len(model.yolo)):
        total_pixel += prediction[i][0].size()[1]
    length_batchsize = (labels[:, 0] * total_pixel).long()
    length_batchsize = torch.concat([length_batchsize for _ in range(anchor_per_layer)])
    truth_arange = torch.concat([torch.arange(len(labels), device = model.device) for i in range(anchor_per_layer)])
    num_layer = {}
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        num_layer[i] = len(anc)
        predicts.append(prediction[i][0])
        mapstride = mapsize[i]**2

        center_truth_xcoord = (((labels[:, 2] / strides[i]) + (labels[:, 2 * 2] / strides[i])) / 2)
        center_truth_x = center_truth_xcoord.long()
        center_truth_x = torch.concat([center_truth_x for _ in range(anchor_per_layer)])
        center_truth_Ycoord = (((labels[:, 3] / strides[i]) + (labels[:, 2 * 2 + 1] / strides[i])) / 2)
        center_truth_y = mapsize[i] * center_truth_Ycoord.long()
        center_truth_y = pre_num + torch.concat([center_truth_y, center_truth_y + mapstride, center_truth_y + mapstride * 2], dim = 0)
        center_truth_y = center_truth_y.long()
    
        truthind = torch.concat([truthind, truth_arange], dim = 0)
        anchorsind = torch.concat([anchorsind, center_truth_x + center_truth_y], dim = 0)
        predictind = torch.concat([predictind, center_truth_x + center_truth_y + length_batchsize], dim = 0)

        pre_num += len(anc)

    del center_truth_x, center_truth_y, center_truth_xcoord, center_truth_Ycoord, prediction, anc, length_batchsize

    anchors = torch.concat(anchors, dim=(0)).float()
    predicts = torch.concat(predicts, dim=(1))
    n, _, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))
    xywh2xyxy(predicts[:, 0:(2*2)], model.imgsize, clamp = False)

    choose_anchors = anchors[anchorsind]
    choose_truths = labels[truthind]

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    iou_anchor_truth = _box_noexpand_iou_(choose_anchors[:, :2*2], choose_truths[:, 2:6])[0]
    pre = 0
    
    index = torch.sort(iou_anchor_truth, descending=True)[1]
    predictind = predictind[index]
    truthind = truthind[index]
    anchorsind = anchorsind[index]
    truth_mark = labels[truthind][:, -1].long()
    del iou_anchor_truth, index

    keepind = []
    tek = set()
    tek2 = set()
    predic = list(predictind.cpu().numpy())
    truth_mark = list(truth_mark.cpu().numpy())
    for i in range(len(predic)):
        ind = predic[i]
        if ind not in tek and truth_mark[i] not in tek2:
            tek.add(ind)
            tek2.add(truth_mark[i])
            keepind.append(i)
    pre = 0
    delta_k = torch.tensor([], dtype=torch.bool).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        delta_mask = box_iou(labels[pre: pre+num, 2:6], predicts[len(anchors) * i : len(anchors) * (i + 1), 0:(2*2)])
        max_val, max_ind = torch.max(delta_mask, dim = 0)
        deltak = max_val > ignore_thresh
        
        delta_k = torch.concat([delta_k, deltak], dim = 0)
        pre += num

    pre_keepind = predictind[keepind]
    choose_predict = predicts[pre_keepind]
    
    # confi_masks_rev = torch.ones(len(predicts), dtype = torch.bool)
    # confi_masks_rev[predictind[keepind]] = False
    # confi_masks_rev[delta_k] = False
    # noconf = predicts[confi_masks_rev, (2*2)].unsqueeze(-1)
    confidence_predict = predicts[:, (2*2)]

    choose_label   = labels[truthind[keepind]]

    del confi_masks_rev, predicts, deltak, delta_k, anchors, anchorsind
    
    if num_scale:
        count_scale = count_scale[choose_label[:, 1].long()]
        count_scale = torch.clamp(count_scale, 0, 3)
    
    ciou, diou, iou, giou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:6]) 
    # iou, _ = _box_noexpand_iou_(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    if num_scale:
        # iou_loss = (1 - ciou) + (1 - diou) + (1 - iou) + (1 - giou) 
        iou_loss = iou_loss * count_scale
    else:
        iou_loss = torch.sum(1 - ciou)
        # iou_loss = (lossx + lossy + lossw + lossh)
        # iou_loss = mseloss[3](choose_predict[:, 0:(2*2)] / model.imgsize, choose_label[:, 2:] / model.imgsize)

    iouloss += iou_loss
    iounow = torch.mean(ciou)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2 * 2 + 1):]  # * choose_predict[:, 2 * 2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    # kk = predicts[confi_masks_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    if num_scale:
        count_scale = torch.unsqueeze(count_scale, dim = -1)
        c_l   += torch.mean(bcecls(classes, class_la) * count_scale)
    else:
        c_l += mseloss[0](classes, class_la)
        # c_l   += torch.sum(torch.square(classes - class_la))
        # c_l += bce0loss[0](classes, class_la)

    # confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    Truth_confid = torch.zeros_like(confidence_predict, dtype = torch.float32, device = model.device)
    Truth_confid[pre_keepind] = 1.0
    if num_scale:
        confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * count_scale) + bce1loss(noconf, torch.zeros_like(noconf))
    else:
        confi_l += mseloss[1](confidence_predict, Truth_confid)
        # confi_l += mseloss[1](confidence, torch.ones_like(confidence)) + mseloss[2](noconf, torch.zeros_like(noconf))
        # confi_l   += torch.sum(torch.square(confidence - torch.ones_like(confidence))) + torch.sum(torch.square(noconf - torch.zeros_like(noconf)))

    # cof = torch.mean(confidence.sigmoid())
    # ncof = torch.mean(noconf.sigmoid())
    cof = torch.mean(confidence_predict[pre_keepind].sigmoid())
    k = confidence_predict.sigmoid()
    ncof = (torch.sum(k) - torch.sum(k[pre_keepind])) / (len(confidence_predict) - len(pre_keepind))
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = c_l + confi_l + iouloss
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, len(labels)


def calculate_losses_darknet(prediction, labels, model, ignore_thresh, \
                            bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    predicts = []
    anchors = []
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    labels = torch.concat([labels, torch.arange(len(labels), device = model.device).unsqueeze(-1)], dim = -1)
    
    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]
    mapsize = [(model.imgsize / i) for i in strides]
    anchor_per_layer = 3
    batchsize = prediction[0][0].size()[0]
    truthind = torch.tensor([], dtype=torch.long).to(model.device)
    anchorsind = torch.tensor([], dtype=torch.long).to(model.device)
    predictind = torch.tensor([], dtype=torch.long).to(model.device)
    
    pre_num = 0
    total_pixel = 0
    for i in range(len(model.yolo)):
        total_pixel += prediction[i][0].size()[1]
    length_batchsize = (labels[:, 0] * total_pixel).long()
    length_batchsize = torch.concat([length_batchsize for _ in range(anchor_per_layer)])
    truth_arange = torch.concat([torch.arange(len(labels), device = model.device) for i in range(anchor_per_layer)])
    num_layer = {}
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        num_layer[i] = len(anc)
        predicts.append(prediction[i][0])
        mapstride = mapsize[i]**2

        center_truth_xcoord = (((labels[:, 2] / strides[i]) + (labels[:, 2 * 2] / strides[i])) / 2)
        center_truth_x = center_truth_xcoord.long()
        center_truth_x = torch.concat([center_truth_x for _ in range(anchor_per_layer)])
        center_truth_Ycoord = (((labels[:, 3] / strides[i]) + (labels[:, 2 * 2 + 1] / strides[i])) / 2)
        center_truth_y = mapsize[i] * center_truth_Ycoord.long()
        center_truth_y = pre_num + torch.concat([center_truth_y, center_truth_y + mapstride, center_truth_y + mapstride * 2], dim = 0)
        center_truth_y = center_truth_y.long()
    
        truthind = torch.concat([truthind, truth_arange], dim = 0)
        anchorsind = torch.concat([anchorsind, center_truth_x + center_truth_y], dim = 0)
        predictind = torch.concat([predictind, center_truth_x + center_truth_y + length_batchsize], dim = 0)

        pre_num += len(anc)

    del center_truth_x, center_truth_y, center_truth_xcoord, center_truth_Ycoord, prediction, anc, length_batchsize

    anchors = torch.concat(anchors, dim=(0)).float()
    predicts = torch.concat(predicts, dim=(1))
    n, _, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))
    xywh2xyxy(predicts[:, 0:(2*2)], model.imgsize, clamp = False)

    choose_anchors = anchors[anchorsind]
    choose_truths = labels[truthind]

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    iou_anchor_truth = _box_noexpand_iou_(choose_anchors[:, :2*2], choose_truths[:, 2:6])[0]
    pre = 0
    
    index = torch.sort(iou_anchor_truth, descending=True)[1]
    predictind = predictind[index]
    truthind = truthind[index]
    anchorsind = anchorsind[index]
    truth_mark = labels[truthind][:, -1].long()
    iou_anchor_truth = iou_anchor_truth[index]
    del iou_anchor_truth, index

    keepind = []
    tek = set()
    tek2 = set()
    predic = list(predictind.cpu().numpy())
    truth_mark = list(truth_mark.cpu().numpy())
    for i in range(len(predic)):
        ind = predic[i]
        if ind not in tek and truth_mark[i] not in tek2:
            tek.add(ind)
            tek2.add(truth_mark[i])
            keepind.append(i)

    # if True:
    #     anchors__ = anchors[anchorsind[keepind]]
    #     xyxy    = labels[truthind[keepind]][:, 2:6]
    #     # pred = predictind[ch]
    #     cvfont = cv2.FONT_HERSHEY_SIMPLEX
    #     for k in range(len(anchors__)):
    #         if k == len(anchors__) - 1:
    #             k = k
    #         # p = pred[k] % total_pixel
    #         tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * (260 - 2*2 - 1)
    #         cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
    #             (int(xyxy[k][2]), int(xyxy[k][3])), (255,126,255), 2)
    #         cv2.rectangle(tmp, (int(anchors__[k][0]), int(anchors__[k][1])), \
    #             (int(anchors__[k][2]), int(anchors__[k][3])), (255,0,0), 1)
    #         cx = ((xyxy[k][0] + xyxy[k][2]) / 2 / 8)
    #         cx2 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 16)
    #         cx3 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 32)
    #         cy = ((xyxy[k][1] + xyxy[k][3]) / 2 / 8)
    #         cy2 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 16)
    #         cy3 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 32)
    #         kk = (cx, cy)
    #         kk2 = (cx2, cy2)
    #         kk3 = (cx3, cy3)
            
    #         acx = ((anchors__[k][0] + anchors__[k][2]) / 2 / 8)
    #         acx2 = ((anchors__[k][0] + anchors__[k][2]) / 2 / 16)
    #         acx3 = ((anchors__[k][0] + xyxy[k][2]) / 2 / 32)
    #         acy = ((anchors__[k][1] + anchors__[k][3]) / 2 / 8)
    #         acy2 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 16)
    #         acy3 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 32)
    #         akk = (acx, acy)
    #         akk2 = (acx2, acy2)
    #         akk3 = (acx3, acy3)
    #         cv2.putText(tmp, str(0), (int(xyxy[k][0]), int(xyxy[k][1]) + 10), cvfont, 0.5, [255, 0, 0], 1)
    #         cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth_%d.jpg'%k, tmp)
    #     exit(0)
        
    pre = 0
    delta_k = torch.tensor([], dtype=torch.bool).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        delta_mask = box_iou(labels[pre: pre+num, 2:6], predicts[len(anchors) * i : len(anchors) * (i + 1), 0:(2*2)])
        max_val, max_ind = torch.max(delta_mask, dim = 0)
        deltak = max_val > ignore_thresh
        
        delta_k = torch.concat([delta_k, deltak], dim = 0)
        pre += num

    pre_keepind = predictind[keepind]
    choose_predict = predicts[pre_keepind]
    
    # confi_masks_rev = torch.ones(len(predicts), dtype = torch.bool)
    # confi_masks_rev[predictind[keepind]] = False
    # confi_masks_rev[delta_k] = False
    # noconf = predicts[confi_masks_rev, (2*2)].unsqueeze(-1)
    confidence_predict = predicts[:, (2*2)]

    choose_label   = labels[truthind[keepind]]

    del confi_masks_rev, predicts, deltak, delta_k
    
    anchor_index = anchorsind[keepind]
    choose_anchor = anchors[anchorsind[keepind]]
    strides = torch.ones_like(anchor_index)
    if len(model.yolo)==2:
        mk = anchor_index < num_layer[0]
        strides[mk] = 16
        strides[~mk] = 32
    elif len(model.yolo)==3:
        mk = anchor_index < num_layer[0]
        strides[mk] = 8
        mk0 = anchor_index < num_layer[1]
        strides[(~mk)&mk0] = 16
        strides[~mk0] = 32
    
    if num_scale:
        count_scale = count_scale[choose_label[:, 1].long()]
        count_scale = torch.clamp(count_scale, 0, 3)

##########################
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]

    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    # for i in torch.unique(maxind):
    #     ch = maxind==i
    #     iounow = iou_scale[ch]
    #     iouch = iounow / torch.max(iounow)
    #     # sum = int(torch.sum(ch))
    #     # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
    #     iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
##########################

    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]

    truthxywh = torch.zeros_like(choose_label[:, 2:6])
    truthxywh[:, :2] = (choose_label[:, 2:2*2] + choose_label[:, 2*2:6]) / 2
    truthxywh[:, 2:] = (choose_label[:, 2*2:6] - choose_label[:, 2:2*2])

    anchorxywh = torch.zeros_like(choose_anchor)
    anchorxywh[:, :2] = (choose_anchor[:, :2] + choose_anchor[:, 2:]) / 2
    anchorxywh[:, :2] /= strides.unsqueeze(-1)
    anchorxywh[:, 2:] = (choose_anchor[:, 2:] - choose_anchor[:, :2])
    
    tx = truthxywh[:, 0] / strides
    tx = tx - anchorxywh[:, 0]
    ty = truthxywh[:, 1] / strides
    ty = ty - anchorxywh[:, 1]
    # if torch.sum(tx > 1) > 0 or torch.sum(ty > 1) or torch.sum(tx > 1) < 0 or torch.sum(ty < 0):
    #     exit(0)
    tw = torch.log(truthxywh[:, 2] / (anchorxywh[:, 2]))
    th = torch.log(truthxywh[:, 3] / (anchorxywh[:, 3]))

    predictxywh = torch.zeros_like(choose_predict[:, 0:2*2])
    predictxywh[:, :2] = (choose_predict[:, :2] + choose_predict[:, 2:2*2]) / 2
    predictxywh[:, 2:] = (choose_predict[:, 2:2*2] - choose_predict[:, :2])

    px = predictxywh[:, 0] / strides
    px = px - anchorxywh[:, 0]
    py = predictxywh[:, 1] / strides
    py = py - anchorxywh[:, 1]
    # if torch.sum(px > 1) > 0 or torch.sum(py > 1) or torch.sum(px > 1) < 0 or torch.sum(py < 0):
    #     exit(0)

    # predictxywh[:, 2][predictxywh[:, 2]==0.0] = 1.0
    # predictxywh[:, 3][predictxywh[:, 3]==0.0] = 1.0
    pw = torch.log(predictxywh[:, 2] / anchorxywh[:, 2])
    ph = torch.log(predictxywh[:, 3] / anchorxywh[:, 3])

    w = truthxywh[:, 2] / model.imgsize
    h = truthxywh[:, 2 + 1] / model.imgsize
    area = h * w
    scale =  2.0 - area
    
    lossx = torch.sum(torch.square(tx - px) * scale)
    lossy = torch.sum(torch.square(ty - py) * scale)
    lossw = torch.sum(torch.square(tw - pw) * scale)
    lossh = torch.sum(torch.square(th - ph) * scale)
    
    # ciou, diou, iou, giou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:]) 
    iou, _ = _box_noexpand_iou_(choose_predict[:, 0:(2*2)], choose_label[:, 2:6])
    # kkk = torch.sum(prediou_!=prediou)
    if num_scale:
        # iou_loss = (1 - ciou) + (1 - diou) + (1 - iou) + (1 - giou) 
        iou_loss = iou_loss * count_scale
    else:
        # iou_loss = torch.sum(1 - ciou)
        iou_loss = (lossx + lossy + lossw + lossh)
        # iou_loss = mseloss[3](choose_predict[:, 0:(2*2)] / model.imgsize, choose_label[:, 2:] / model.imgsize)

    iouloss += iou_loss
    iounow = torch.mean(iou)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2 * 2 + 1):] * choose_predict[:, 2 * 2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    # kk = predicts[confi_masks_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    if num_scale:
        count_scale = torch.unsqueeze(count_scale, dim = -1)
        c_l   += torch.mean(bcecls(classes, class_la) * count_scale)
    else:
        c_l += mseloss[0](classes, class_la)
        # c_l   += torch.sum(torch.square(classes - class_la))

    # confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    Truth_confid = torch.zeros_like(confidence_predict, dtype = torch.float32, device = model.device)
    Truth_confid[pre_keepind] = 1.0
    if num_scale:
        confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * count_scale) + bce1loss(noconf, torch.zeros_like(noconf))
    else:
        confi_l += mseloss[1](confidence_predict, Truth_confid)
        # confi_l += mseloss[1](confidence, torch.ones_like(confidence)) + mseloss[2](noconf, torch.zeros_like(noconf))
        # confi_l   += torch.sum(torch.square(confidence - torch.ones_like(confidence))) + torch.sum(torch.square(noconf - torch.zeros_like(noconf)))

    # cof = torch.mean(confidence.sigmoid())
    # ncof = torch.mean(noconf.sigmoid())
    cof = torch.mean(confidence_predict[pre_keepind].sigmoid())
    k = confidence_predict.sigmoid()
    ncof = (torch.sum(k) - torch.sum(k[pre_keepind])) / (len(confidence_predict) - len(pre_keepind))
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = c_l + confi_l + iouloss
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, len(labels)

def calculate_losses_Alexeydarknet(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                    bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    predicts = []
    anchors = []
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)

    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]
    mapsize = [(model.imgsize / i) for i in strides]
    anchor_per_layer = 3
    batchsize = prediction[0][0].size()[0]
    truthind = torch.tensor([], dtype=torch.long).to(model.device)
    anchorsind = torch.tensor([], dtype=torch.long).to(model.device)
    predictind = torch.tensor([], dtype=torch.long).to(model.device)
    
    pre_num = 0
    total_pixel = 0
    for i in range(len(model.yolo)):
        total_pixel += prediction[i][0].size()[1]
    length_batchsize = (labels[:, 0] * total_pixel).long()
    length_batchsize = torch.concat([length_batchsize for _ in range(anchor_per_layer)])
    truth_arange = torch.concat([torch.arange(len(labels), device = model.device) for i in range(anchor_per_layer)])
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        predicts.append(prediction[i][0])
        mapstride = mapsize[i]**2

        center_truth_xcoord = (((labels[:, 2] / strides[i]) + (labels[:, 2 * 2] / strides[i])) / 2)
        center_truth_x = center_truth_xcoord.long()
        center_truth_x = torch.concat([center_truth_x for _ in range(anchor_per_layer)])
        center_truth_Ycoord = (((labels[:, 3] / strides[i]) + (labels[:, 2 * 2 + 1] / strides[i])) / 2)
        center_truth_y = mapsize[i] * center_truth_Ycoord.long()
        center_truth_y = pre_num + torch.concat([center_truth_y, center_truth_y + mapstride, center_truth_y + mapstride * 2], dim = 0)
        center_truth_y = center_truth_y.long()
    
        truthind = torch.concat([truthind, truth_arange], dim = 0)
        anchorsind = torch.concat([anchorsind, center_truth_x + center_truth_y], dim = 0)
        predictind = torch.concat([predictind, center_truth_x + center_truth_y + length_batchsize], dim = 0)

        pre_num += len(anc)

    del center_truth_x, center_truth_y, center_truth_xcoord, center_truth_Ycoord, prediction, anc, length_batchsize

    anchors = torch.concat(anchors, dim=(0)).float()
    predicts = torch.concat(predicts, dim=(1))
    n, _, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))
    xywh2xyxy(predicts[:, 0:(2*2)], model.imgsize, clamp = False)

    choose_anchors = anchors[anchorsind]
    choose_truths = labels[truthind]

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    iou_anchor_truth = _box_noexpand_iou_(choose_anchors[:, :2*2], choose_truths[:, 2:6])[0]
    mask = iou_anchor_truth > iou_thresh
    choose_truths = choose_truths[mask]
    predictind = predictind[mask]
    iou_anchor_truth = iou_anchor_truth[mask]
    truthind = truthind[mask]
    del choose_anchors
    pre = 0
    
    index = torch.sort(iou_anchor_truth, descending=True)[1]
    predictind = predictind[index]
    truthind = truthind[index]
    del iou_anchor_truth, index

    keepind = []
    tek = set()
    predic = list(predictind.cpu().numpy())
    for i in range(len(predic)):
        ind = predic[i]
        if ind not in tek:
            tek.add(ind)
            keepind.append(i)
    pre = 0
    delta_k = torch.tensor([], dtype=torch.bool).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        delta_mask = box_iou(labels[pre: pre+num, 2:], predicts[len(anchors) * i : len(anchors) * (i + 1), 0:(2*2)])
        max_val, max_ind = torch.max(delta_mask, dim = 0)
        deltak = max_val > ignore_thresh
        
        delta_k = torch.concat([delta_k, deltak], dim = 0)
        pre += num
    pre_keepind = predictind[keepind]
    choose_predict = predicts[pre_keepind]
    confidence_predict = predicts[:, (2*2)]
    # confi_masks_rev = torch.ones(len(predicts), dtype = torch.bool)
    # confi_masks_rev[pre_keepind] = False
    # confi_masks_rev[delta_k] = False
    # noconf = predicts[confi_masks_rev, (2*2)].unsqueeze(-1)

    choose_label   = labels[truthind[keepind], :]

    del confi_masks_rev, predicts, deltak, delta_k

    if num_scale:
        count_scale = count_scale[choose_label[:, 1].long()]
        count_scale = torch.clamp(count_scale, 0, 6)

##########################
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]

    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    # for i in torch.unique(maxind):
    #     ch = maxind==i
    #     iounow = iou_scale[ch]
    #     iouch = iounow / torch.max(iounow)
    #     # sum = int(torch.sum(ch))
    #     # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
    #     iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
##########################

    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    # ciou, diou, iou, giou = complete_box_iou_no_expand(choose_iou_predict[:, 0:(2*2)], choose_iou_label[:, 2:])
    ciou, diou, iou, giou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # del choose_iou_predict, choose_iou_label
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))
    if num_scale:
        iou_loss = (1 - ciou) + (1 - diou) + (1 - iou) + (1 - giou) 
        iou_loss = iou_loss * count_scale
    else:
        iou_loss = (1 - ciou)         # * iou_scale         # scale * count_scale
        # iou_loss = mseloss(choose_predict[:, 0:(2*2)] / model.imgsize, choose_label[:, 2:] / model.imgsize)

    iou_loss = iou_loss # / one_anchor_multilabel
    iouloss += torch.sum(iou_loss)
    iounow = torch.mean(ciou)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2*2+1):]  # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32, device = model.device) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    # kk = predicts[confi_masks_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    if num_scale:
        count_scale = torch.unsqueeze(count_scale, dim = -1)
        c_l   += torch.mean(bcecls(classes, class_la) * count_scale)
    else:
        # c_l   += torch.sum(torch.square(classes - class_la))
        # c_l   += mseloss[0](classes, class_la)
        assert bce0loss.reduction=='sum'
        c_l   += bce0loss(classes, class_la)
        # sigmoid = 1 / (1+torch.exp(-classes))
        # count_scale = torch.unsqueeze(count_scale, dim = -1)
        # c_l += -torch.sum(((class_la * np.log(sigmoid + 1e-10)) + ((1 - class_la) * np.log((1 - sigmoid) + 1e-10))) * count_scale)

    Truth_confid = torch.zeros_like(confidence_predict, dtype = torch.float32, device = model.device)
    Truth_confid[pre_keepind] = 1.0 # ciou.detach().clamp(0).type(Truth_confid.dtype)
    if num_scale:
        confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * count_scale) + bce1loss(noconf, torch.zeros_like(noconf))
    else:
        # confi_l   += torch.sum(torch.square(confidence - torch.ones_like(confidence))) + torch.sum(torch.square(noconf - torch.zeros_like(noconf)))
        # confi_l   += mseloss[1](confidence, torch.ones_like(confidence)) + mseloss[2](noconf, torch.zeros_like(noconf))
        assert bce1loss.reduction=='sum'
        confi_l   += bce1loss(confidence_predict, Truth_confid) #+ bce2loss(noconf, torch.zeros_like(noconf))
        # sigmoid = 1 / (1+torch.exp(-confidence))
        # confi_l += -torch.sum(((torch.ones_like(confidence) * np.log(sigmoid + 1e-10)) + ((1 - torch.ones_like(confidence)) * np.log((1 - sigmoid) + 1e-10))) * count_scale)
        # confi_l   += bce2loss(noconf, torch.zeros_like(noconf))

    cof = torch.mean(confidence_predict[pre_keepind].sigmoid())
    k = confidence_predict.sigmoid()
    ncof = (torch.sum(k) - torch.sum(k[pre_keepind])) / (len(confidence_predict) - len(pre_keepind))
    # cof = torch.mean(confidence.sigmoid())
    # ncof = torch.mean(noconf.sigmoid())
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = c_l + confi_l + iouloss
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, len(labels)

def calculate_losses_yolofive_revise(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                    bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):

    predicts = []
    anchors = []
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    
    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]
    mapsize = [(model.imgsize / i) for i in strides]
    anchor_per_layer = 3
    batchsize = prediction[0][0].size()[0]
    truthind = torch.tensor([], dtype=torch.long).to(model.device)
    anchorsind = torch.tensor([], dtype=torch.long).to(model.device)
    predictind = torch.tensor([], dtype=torch.long).to(model.device)
    
    pre_num = 0
    total_pixel = 0
    for i in range(len(model.yolo)):
        total_pixel += prediction[i][0].size()[1]
    length_batchsize = (labels[:, 0] * total_pixel).long()
    length_batchsize = torch.concat([length_batchsize for _ in range(anchor_per_layer)])
    truth_arange = torch.concat([torch.arange(len(labels), device = model.device) for i in range(anchor_per_layer)])
    num_layer = {}
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        num_layer[i] = len(anc)
        predicts.append(prediction[i][0])
        mapstride = mapsize[i]**2
        
######### 0, 0
        center_truth_xcoord = (((labels[:, 2] / strides[i]) + (labels[:, 2 * 2] / strides[i])) / 2)
        center_truth_x = center_truth_xcoord.long()
        center_truth_x = torch.concat([center_truth_x for _ in range(anchor_per_layer)])
        center_truth_Ycoord = (((labels[:, 3] / strides[i]) + (labels[:, 2 * 2 + 1] / strides[i])) / 2)
        center_truth_y = mapsize[i] * center_truth_Ycoord.long()
        center_truth_y = pre_num + torch.concat([center_truth_y, center_truth_y + mapstride, center_truth_y + mapstride * 2], dim = 0)
        center_truth_y = center_truth_y.long()

        truthind = torch.concat([truthind, truth_arange], dim = 0)
        anchorsind = torch.concat([anchorsind, center_truth_x + center_truth_y], dim = 0)
        predictind = torch.concat([predictind, center_truth_x + center_truth_y + length_batchsize], dim = 0)

######### -1/2, 0
        chk = ((center_truth_xcoord%1) < (3/6)) & (center_truth_xcoord > 1)
        center_truth_x_left = center_truth_xcoord.clone()
        center_truth_x_left[~chk] = -666666
        center_truth_x_left[chk]  = center_truth_x_left[chk] - 3/6
        chk_left_x = torch.concat([chk for _ in range(anchor_per_layer)])
        center_truth_x_left = torch.concat([center_truth_x_left for _ in range(anchor_per_layer)]).long()
        truthind = torch.concat([truthind, truth_arange[chk_left_x]], dim = 0)
        anchorsind = torch.concat([anchorsind, (center_truth_x_left + center_truth_y)[chk_left_x]], dim = 0)
        predictind = torch.concat([predictind, (center_truth_x_left + center_truth_y + length_batchsize)[chk_left_x]], dim = 0)

######### +1/2, 0
        rever = mapsize[i] - center_truth_xcoord
        reversed_chk = ((rever % 1) < (3/6)) & (rever > 1)
        center_truth_x_right = center_truth_xcoord.clone()
        center_truth_x_right[~reversed_chk] = -666666
        center_truth_x_right[reversed_chk]  = center_truth_x_right[reversed_chk] + 3/6
        reversed_chk_right_x = torch.concat([reversed_chk for _ in range(anchor_per_layer)])
        center_truth_x_right = torch.concat([center_truth_x_right for _ in range(anchor_per_layer)]).long()
        truthind = torch.concat([truthind, truth_arange[reversed_chk_right_x]], dim = 0)
        anchorsind = torch.concat([anchorsind, (center_truth_x_right + center_truth_y)[reversed_chk_right_x]], dim = 0)
        predictind = torch.concat([predictind, (center_truth_x_right + center_truth_y + length_batchsize)[reversed_chk_right_x]], dim = 0)
######### [0, -1/2]; [-1/2, -1/2], [+1/2, -1/2]
        chk = ((center_truth_Ycoord%1) < (3/6)) & (center_truth_Ycoord > 1)
        center_truth_y_top = center_truth_Ycoord.clone()
        center_truth_y_top[~chk] = -666666
        center_truth_y_top[chk]  = center_truth_y_top[chk] - 3/6
        center_truth_y_top = mapsize[i] * center_truth_y_top.long()
        center_truth_y_top = pre_num + torch.concat([center_truth_y_top, center_truth_y_top + mapstride, center_truth_y_top + mapstride * 2], dim = 0)
        center_truth_y_top = center_truth_y_top.long()
        chk_top_y = torch.concat([chk for _ in range(anchor_per_layer)])

        truth_choose = truth_arange[chk_top_y]
        k0 = center_truth_x + center_truth_y_top
        k1 = center_truth_x_left + center_truth_y_top
        k2 = center_truth_x_right + center_truth_y_top
        bool0 = chk_left_x & chk_top_y
        bool2 = reversed_chk_right_x & chk_top_y
        truthind = torch.concat([truthind, truth_arange[chk_top_y]], dim = 0)
        # truthind = torch.concat([truthind, truth_arange[chk_top_y], truth_arange[bool0], truth_arange[bool2]], dim = 0)
        anchorsind = torch.concat([anchorsind, k0[chk_top_y]], dim = 0)
        # anchorsind = torch.concat([anchorsind, k1[bool0]], dim = 0)
        # anchorsind = torch.concat([anchorsind, k2[bool2]], dim = 0)
        predictind = torch.concat([predictind, (k0 + length_batchsize)[chk_top_y]], dim = 0)
        # predictind = torch.concat([predictind, (k1 + length_batchsize)[bool0]], dim = 0)
        # predictind = torch.concat([predictind, (k2 + length_batchsize)[bool2]], dim = 0)
######### [0, +1/2]; [-1/2, +1/2], [+1/2, +1/2]
        rever = mapsize[i] - center_truth_Ycoord
        reversed_chk = ((rever % 1) < (3/6)) & (rever > 1)
        center_truth_y_bottom = center_truth_Ycoord.clone()
        center_truth_y_bottom[~reversed_chk] = -666666
        center_truth_y_bottom[reversed_chk]  = center_truth_y_bottom[reversed_chk] + 3/6
        center_truth_y_bottom = mapsize[i] * center_truth_y_bottom.long()
        center_truth_y_bottom = pre_num + torch.concat([center_truth_y_bottom, center_truth_y_bottom + mapstride, center_truth_y_bottom + mapstride * 2], dim = 0)
        center_truth_y_bottom = center_truth_y_bottom.long()
        reversed_chk_bottom_y = torch.concat([reversed_chk for _ in range(anchor_per_layer)])

        truth_choose = truth_arange[reversed_chk_bottom_y]
        k0 = center_truth_x + center_truth_y_bottom
        k1 = center_truth_x_left + center_truth_y_bottom
        k2 = center_truth_x_right + center_truth_y_bottom
        bool0 = chk_left_x & reversed_chk_bottom_y
        bool2 = reversed_chk_right_x & reversed_chk_bottom_y
        truthind = torch.concat([truthind, truth_arange[reversed_chk_bottom_y]], dim = 0)
        # truthind = torch.concat([truthind, truth_arange[reversed_chk_bottom_y], truth_arange[bool0], truth_arange[bool2]], dim = 0)
        anchorsind = torch.concat([anchorsind, k0[reversed_chk_bottom_y]], dim = 0)
        # anchorsind = torch.concat([anchorsind, k1[bool0]], dim = 0)
        # anchorsind = torch.concat([anchorsind, k2[bool2]], dim = 0)
        predictind = torch.concat([predictind, (k0 + length_batchsize)[reversed_chk_bottom_y]], dim = 0)
        # predictind = torch.concat([predictind, (k1 + length_batchsize)[bool0]], dim = 0)
        # predictind = torch.concat([predictind, (k2 + length_batchsize)[bool2]], dim = 0)

        pre_num += len(anc)
########

    del center_truth_x, center_truth_y, center_truth_xcoord, center_truth_Ycoord, rever, center_truth_x_left, \
        chk, chk_left_x, reversed_chk, center_truth_x_right, reversed_chk_right_x, center_truth_y_top, \
        chk_top_y, truth_choose, center_truth_y_bottom, reversed_chk_bottom_y, anc, length_batchsize, \
        k0, k1, k2, bool0, bool2 #prediction
    anchors = torch.concat(anchors, dim=(0)).float()
    predicts = torch.concat(predicts, dim=(1))
    n, _, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    choose_anchors = anchors[anchorsind]
    choose_truths = labels[truthind]

    label_width  = choose_truths[:, 2*2] - choose_truths[:, 2]
    label_height = choose_truths[:, 2*2+1] - choose_truths[:, 2+1]
    A_width  = choose_anchors[:, 2] - choose_anchors[:, 0]
    A_height = choose_anchors[:, 3] - choose_anchors[:, 1]
    max_width = torch.max(label_width/A_width, A_width/label_width)
    max_height = torch.max(label_height/A_height, A_height/label_height)
    ratio = torch.max(max_height, max_width) < (2*2)

    choose_anchors = choose_anchors[ratio]
    choose_truths = choose_truths[ratio]
    predictind = predictind[ratio]
    truthind = truthind[ratio]
    del label_width, label_height, A_width, A_height, max_width, max_height, ratio, anchors, anchorsind
    
    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    iou_anchor_truth = _box_noexpand_iou_(choose_anchors[:, :2*2], choose_truths[:, 2:6])[0]
    
    # if True:
    #     ch = choose_truths[:, 0].long()==0
    #     anchors__ = choose_anchors[ch]
    #     xyxy    = choose_truths[ch][:, 2:6]
    #     pred = predictind[ch]
    #     cvfont = cv2.FONT_HERSHEY_SIMPLEX
    #     for k in range(len(anchors__)):
    #         if k == len(anchors__) - 1:
    #             k = k
    #         p = pred[k] % total_pixel
    #         tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * (260 - 2*2 - 1)
    #         cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
    #             (int(xyxy[k][2]), int(xyxy[k][3])), (255,126,255), 2)
    #         cv2.rectangle(tmp, (int(anchors__[k][0]), int(anchors__[k][1])), \
    #             (int(anchors__[k][2]), int(anchors__[k][3])), (255,0,0), 1)
    #         cx = ((xyxy[k][0] + xyxy[k][2]) / 2 / 8)
    #         cx2 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 16)
    #         cx3 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 32)
    #         cy = ((xyxy[k][1] + xyxy[k][3]) / 2 / 8)
    #         cy2 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 16)
    #         cy3 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 32)
    #         kk = (cx, cy)
    #         kk2 = (cx2, cy2)
    #         kk3 = (cx3, cy3)
            
    #         acx = ((anchors__[k][0] + anchors__[k][2]) / 2 / 8)
    #         acx2 = ((anchors__[k][0] + anchors__[k][2]) / 2 / 16)
    #         acx3 = ((anchors__[k][0] + xyxy[k][2]) / 2 / 32)
    #         acy = ((anchors__[k][1] + anchors__[k][3]) / 2 / 8)
    #         acy2 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 16)
    #         acy3 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 32)
    #         akk = (acx, acy)
    #         akk2 = (acx2, acy2)
    #         akk3 = (acx3, acy3)
    #         cv2.putText(tmp, str(ch[k]), (int(xyxy[k][0]), int(xyxy[k][1]) + 10), cvfont, 0.5, [255, 0, 0], 1)
    #         cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth_%d.jpg'%k, tmp)
    #     exit(0)
        
    del choose_anchors
    
    index = torch.sort(iou_anchor_truth, descending=True)[1]
    predictind = predictind[index]
    truthind = truthind[index]
    del iou_anchor_truth, index

    keepind = []
    tek = set()
    # predictind = predictind.cpu().numpy()
    predic = list(predictind.cpu().numpy())
    for i in range(len(predic)):
        ind = predic[i]
        if ind not in tek:
            tek.add(ind)
            keepind.append(i)
    pre_keepind = predictind[keepind]
    choose_predict = predicts[pre_keepind]

    choose_label   = labels[truthind[keepind], :]

    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)

    # # #
    # indkeep = pre_keepind % total_pixel
    # layer_scale = torch.ones(len(pre_keepind), dtype = torch.float32, device = model.device)
    # if len(model.yolo)==3:
    #     kid = indkeep < num_layer[0]
    #     # k = torch.sum(kid)
    #     layer_scale[kid] = 1 / torch.sum(kid)
    #     kidkid = (indkeep < (num_layer[1] + num_layer[0]))
    #     kl = kidkid & (~kid)
    #     # kk = torch.sum(kl)
    #     layer_scale[kl] = 1 / torch.sum(kl)
    #     kidkid = ~kidkid
    #     # end = torch.sum(kidkid)
    #     layer_scale[kidkid] = 1 / torch.sum(kidkid)
    #     del kid, kl, kidkid
    # else:
    #     kid = indkeep < num_layer[0]
    #     layer_scale[kid] = 1 / torch.sum(kid)
    #     kid = ~kid
    #     layer_scale[kid] = 1 / torch.sum(kid)
    #     del kid
    # # #
    # confi_masks_rev = torch.ones(len(predicts), dtype = torch.bool)
    # confi_masks_rev[predictind[keepind]] = False
    # noconf = predicts[confi_masks_rev, (2*2)].unsqueeze(-1)
    confidence_predict = predicts[:, (2*2)]
    del predicts
    
    if num_scale:
        count_scale = count_scale[choose_label[:, 1].long()]
        count_scale = torch.clamp(count_scale, 0, 3)

    ciou, diou, iou, giou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))

    iou_loss = (1 - ciou)    # * layer_scale  # scale * count_scale

    # iou_loss = iou_loss                             # / one_anchor_multilabel
    iouloss = torch.sum(iou_loss)   #/ (len(ciou) / len(model.yolo))
    iounow = torch.mean(ciou)
    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2*2+1):] # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32, device = model.device) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    # kk = predicts[confi_masks_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    num_classes = classes.size(1)
    if num_scale:
        count_scale = torch.unsqueeze(count_scale, dim = -1)
        c_l   = torch.mean(bcecls(classes, class_la) * count_scale)
    else:
        # c_l   += torch.mean(torch.pow(classes - class_la, 2))
        # layer_scale = layer_scale.unsqueeze(-1)
        # c_l   = torch.sum(bce0loss(classes, class_la) * (layer_scale / num_classes))  # / (len(classes) / len(model.yolo)) # + bceloss(kk, torch.zeros_like(kk))
        assert bce0loss.reduction=='sum'
        c_l   = bce0loss(classes, class_la)

    Truth_confid = torch.zeros_like(confidence_predict, dtype = torch.float32, device = model.device)
    Truth_confid[pre_keepind] = 1.0 #ciou.detach().clamp(0).type(Truth_confid.dtype)
    # confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    if num_scale:
        pass
        # confi_l   = torch.mean(bcecof(confidence, torch.ones_like(confidence)) * count_scale) + bce1loss(noconf, torch.zeros_like(noconf))
    else:
        # confi_l   += torch.mean(torch.pow(confidence - torch.ones_like(confidence), 2)) +  torch.mean(torch.pow(noconf - torch.zeros_like(noconf), 2))
        # ciou = ciou.unsqueeze(-1)
        # cofobj = torch.ones_like(confidence) * ciou.clamp(0).type(confidence.dtype)
        # confi_l   += bcecof(confidence, cofobj) + bce2loss(noconf, torch.zeros_like(noconf))
        assert bce1loss.reduction=='sum'
        confi_l   = bce1loss(confidence_predict, Truth_confid) #/ len(confidence_predict)
        # confi_l   = (bce1loss(confidence, torch.ones_like(confidence)) + bce2loss(noconf, torch.zeros_like(noconf))) 

    # hypbox = 0.06 * 3 / len(model.yolo)  # scale to layers
    # hypcls = 0.6 * num_classes / 80 * 3 / len(model.yolo)  # scale to classes and layers
    # hypobj = 1.0 * (model.imgsize / 640) ** 2 * 3 / len(model.yolo)  # scale to image size and layers

    # confi_l *= hypobj
    # c_l *= hypcls
    # iouloss *= hypbox
    
    # confi_l *= 1.0
    # c_l *= 0.6
    # iouloss *= 0.06

    cof = torch.mean(confidence_predict[pre_keepind].sigmoid())
    k = confidence_predict.sigmoid()
    ncof = (torch.sum(k) - torch.sum(k[pre_keepind])) / (len(confidence_predict) - len(pre_keepind))
    # cof = torch.mean(confidence.sigmoid())
    # ncof = torch.mean(noconf.sigmoid())
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = (c_l + confi_l + iouloss) #* batchsize
    del ciou, diou, iou, giou
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, len(labels)

def crop_mask(masks, boxes):
    #  https://github.com/ultralytics/yolov5/tree/master/utils
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [n, h, w] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    #  https://github.com/ultralytics/yolov5/tree/master/utils
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = (masks_in @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

def masks2segments(masks, strategy='largest'):
    #  https://github.com/ultralytics/yolov5/tree/master/utils
    # Convert masks(n,160,160) into segments(n,xy)
    segments = []
    for x in masks.int().cpu().numpy().astype('uint8'):
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if c:
            if strategy == 'concat':  # concatenate all segments
                c = np.concatenate([x.reshape(-1, 2) for x in c])
            elif strategy == 'largest':  # select largest segment
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:
            c = np.zeros((0, 2))  # no segments found
        segments.append(c.astype('float32'))
    return segments

def calculate_losses_yolofive(prediction, labels, model, ignore_thresh, iou_thresh, count_scale, \
                                    bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    prototype = prediction[-1]
    prediction = prediction[:-1]
    predicts = []
    masks = []
    anchors = []
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    
    if len(model.yolo)==2:
        strides = [16, 32]
    else:
        strides = [8, 16, 32]
    mapsize = [(model.imgsize / i) for i in strides]
    anchor_per_layer = 3
    batchsize = prediction[0][0].size()[0]
    truthind = torch.tensor([], dtype=torch.long).to(model.device)
    anchorsind = torch.tensor([], dtype=torch.long).to(model.device)
    predictind = torch.tensor([], dtype=torch.long).to(model.device)
    
    pre_num = 0
    total_pixel = 0
    for i in range(len(model.yolo)):
        total_pixel += prediction[i][0].size()[1]
    length_batchsize = (labels[:, 0] * total_pixel).long()
    length_batchsize = torch.concat([length_batchsize for _ in range(anchor_per_layer)])
    truth_arange = torch.concat([torch.arange(len(labels), device = model.device) for i in range(anchor_per_layer)])
    num_layer = {}
    for i in range(len(model.yolo)):
        anc = prediction[i][1]
        anchors.append(anc)
        num_layer[i] = len(anc)
        predicts.append(prediction[i][0][:, :, :(model.num_classes + 1 + 2 + 2)])
        masks.append(prediction[i][0][:, :, (model.num_classes + 1 + 2 + 2):])
        mapstride = mapsize[i]**2
        
######### 0, 0
        center_truth_xcoord = (((labels[:, 2] / strides[i]) + (labels[:, 2 * 2] / strides[i])) / 2)
        center_truth_x = center_truth_xcoord.long()
        center_truth_x = torch.concat([center_truth_x for _ in range(anchor_per_layer)])
        center_truth_Ycoord = (((labels[:, 3] / strides[i]) + (labels[:, 2 * 2 + 1] / strides[i])) / 2)
        center_truth_y = mapsize[i] * center_truth_Ycoord.long()
        center_truth_y = pre_num + torch.concat([center_truth_y, center_truth_y + mapstride, center_truth_y + mapstride * 2], dim = 0)
        center_truth_y = center_truth_y.long()

        truthind = torch.concat([truthind, truth_arange], dim = 0)
        anchorsind = torch.concat([anchorsind, center_truth_x + center_truth_y], dim = 0)
        predictind = torch.concat([predictind, center_truth_x + center_truth_y + length_batchsize], dim = 0)

######### -1/2, 0
        chk = ((center_truth_xcoord%1) < (3/6)) & (center_truth_xcoord > 1)
        center_truth_x_left = center_truth_xcoord.clone()
        center_truth_x_left[~chk] = -666666
        center_truth_x_left[chk]  = center_truth_x_left[chk] - 3/6
        chk_left_x = torch.concat([chk for _ in range(anchor_per_layer)])
        center_truth_x_left = torch.concat([center_truth_x_left for _ in range(anchor_per_layer)]).long()
        truthind = torch.concat([truthind, truth_arange[chk_left_x]], dim = 0)
        anchorsind = torch.concat([anchorsind, (center_truth_x_left + center_truth_y)[chk_left_x]], dim = 0)
        predictind = torch.concat([predictind, (center_truth_x_left + center_truth_y + length_batchsize)[chk_left_x]], dim = 0)

######### +1/2, 0
        rever = mapsize[i] - center_truth_xcoord
        reversed_chk = ((rever % 1) < (3/6)) & (rever > 1)
        center_truth_x_right = center_truth_xcoord.clone()
        center_truth_x_right[~reversed_chk] = -666666
        center_truth_x_right[reversed_chk]  = center_truth_x_right[reversed_chk] + 3/6
        reversed_chk_right_x = torch.concat([reversed_chk for _ in range(anchor_per_layer)])
        center_truth_x_right = torch.concat([center_truth_x_right for _ in range(anchor_per_layer)]).long()
        truthind = torch.concat([truthind, truth_arange[reversed_chk_right_x]], dim = 0)
        anchorsind = torch.concat([anchorsind, (center_truth_x_right + center_truth_y)[reversed_chk_right_x]], dim = 0)
        predictind = torch.concat([predictind, (center_truth_x_right + center_truth_y + length_batchsize)[reversed_chk_right_x]], dim = 0)
######### [0, -1/2]; [-1/2, -1/2], [+1/2, -1/2]
        chk = ((center_truth_Ycoord%1) < (3/6)) & (center_truth_Ycoord > 1)
        center_truth_y_top = center_truth_Ycoord.clone()
        center_truth_y_top[~chk] = -666666
        center_truth_y_top[chk]  = center_truth_y_top[chk] - 3/6
        center_truth_y_top = mapsize[i] * center_truth_y_top.long()
        center_truth_y_top = pre_num + torch.concat([center_truth_y_top, center_truth_y_top + mapstride, center_truth_y_top + mapstride * 2], dim = 0)
        center_truth_y_top = center_truth_y_top.long()
        chk_top_y = torch.concat([chk for _ in range(anchor_per_layer)])

        truth_choose = truth_arange[chk_top_y]
        k0 = center_truth_x + center_truth_y_top
        k1 = center_truth_x_left + center_truth_y_top
        k2 = center_truth_x_right + center_truth_y_top
        bool0 = chk_left_x & chk_top_y
        bool2 = reversed_chk_right_x & chk_top_y
        # truthind = torch.concat([truthind, truth_arange[chk_top_y]], dim = 0)
        truthind = torch.concat([truthind, truth_arange[chk_top_y], truth_arange[bool0], truth_arange[bool2]], dim = 0)
        anchorsind = torch.concat([anchorsind, k0[chk_top_y]], dim = 0)
        anchorsind = torch.concat([anchorsind, k1[bool0]], dim = 0)
        anchorsind = torch.concat([anchorsind, k2[bool2]], dim = 0)
        predictind = torch.concat([predictind, (k0 + length_batchsize)[chk_top_y]], dim = 0)
        predictind = torch.concat([predictind, (k1 + length_batchsize)[bool0]], dim = 0)
        predictind = torch.concat([predictind, (k2 + length_batchsize)[bool2]], dim = 0)
######### [0, +1/2]; [-1/2, +1/2], [+1/2, +1/2]
        rever = mapsize[i] - center_truth_Ycoord
        reversed_chk = ((rever % 1) < (3/6)) & (rever > 1)
        center_truth_y_bottom = center_truth_Ycoord.clone()
        center_truth_y_bottom[~reversed_chk] = -666666
        center_truth_y_bottom[reversed_chk]  = center_truth_y_bottom[reversed_chk] + 3/6
        center_truth_y_bottom = mapsize[i] * center_truth_y_bottom.long()
        center_truth_y_bottom = pre_num + torch.concat([center_truth_y_bottom, center_truth_y_bottom + mapstride, center_truth_y_bottom + mapstride * 2], dim = 0)
        center_truth_y_bottom = center_truth_y_bottom.long()
        reversed_chk_bottom_y = torch.concat([reversed_chk for _ in range(anchor_per_layer)])

        truth_choose = truth_arange[reversed_chk_bottom_y]
        k0 = center_truth_x + center_truth_y_bottom
        k1 = center_truth_x_left + center_truth_y_bottom
        k2 = center_truth_x_right + center_truth_y_bottom
        bool0 = chk_left_x & reversed_chk_bottom_y
        bool2 = reversed_chk_right_x & reversed_chk_bottom_y
        # truthind = torch.concat([truthind, truth_arange[reversed_chk_bottom_y]], dim = 0)
        truthind = torch.concat([truthind, truth_arange[reversed_chk_bottom_y], truth_arange[bool0], truth_arange[bool2]], dim = 0)
        anchorsind = torch.concat([anchorsind, k0[reversed_chk_bottom_y]], dim = 0)
        anchorsind = torch.concat([anchorsind, k1[bool0]], dim = 0)
        anchorsind = torch.concat([anchorsind, k2[bool2]], dim = 0)
        predictind = torch.concat([predictind, (k0 + length_batchsize)[reversed_chk_bottom_y]], dim = 0)
        predictind = torch.concat([predictind, (k1 + length_batchsize)[bool0]], dim = 0)
        predictind = torch.concat([predictind, (k2 + length_batchsize)[bool2]], dim = 0)

        pre_num += len(anc)
########

    del center_truth_x, center_truth_y, center_truth_xcoord, center_truth_Ycoord, rever, center_truth_x_left, \
        chk, chk_left_x, reversed_chk, center_truth_x_right, reversed_chk_right_x, center_truth_y_top, \
        chk_top_y, truth_choose, center_truth_y_bottom, reversed_chk_bottom_y, anc, length_batchsize, \
        k0, k1, k2, bool0, bool2 #prediction
    anchors = torch.concat(anchors, dim=(0)).float()
    predicts = torch.concat(predicts, dim=(1))
    masks = torch.concat(masks, dim=(1))
    n, _, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))
    n, _, kl = masks.size()
    masks = torch.reshape(masks, (-1, kl))

    choose_anchors = anchors[anchorsind]
    choose_truths = labels[truthind]

    label_width  = choose_truths[:, 2*2] - choose_truths[:, 2]
    label_height = choose_truths[:, 2*2+1] - choose_truths[:, 2+1]
    A_width  = choose_anchors[:, 2] - choose_anchors[:, 0]
    A_height = choose_anchors[:, 3] - choose_anchors[:, 1]
    max_width = torch.max(label_width/A_width, A_width/label_width)
    max_height = torch.max(label_height/A_height, A_height/label_height)
    ratio = torch.max(max_height, max_width) < (2*2)

    choose_anchors = choose_anchors[ratio]
    choose_truths = choose_truths[ratio]
    predictind = predictind[ratio]
    truthind = truthind[ratio]
    del label_width, label_height, A_width, A_height, max_width, max_height, ratio, anchors, anchorsind
    
    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)
    maskloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    iou_anchor_truth = _box_noexpand_iou_(choose_anchors[:, :2*2], choose_truths[:, 2:6])[0]
    
    # if True:
    #     ch = choose_truths[:, 0].long()==0
    #     anchors__ = choose_anchors[ch]
    #     xyxy    = choose_truths[ch][:, 2:6]
    #     pred = predictind[ch]
    #     cvfont = cv2.FONT_HERSHEY_SIMPLEX
    #     for k in range(len(anchors__)):
    #         if k == len(anchors__) - 1:
    #             k = k
    #         p = pred[k] % total_pixel
    #         tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * (260 - 2*2 - 1)
    #         cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
    #             (int(xyxy[k][2]), int(xyxy[k][3])), (255,126,255), 2)
    #         cv2.rectangle(tmp, (int(anchors__[k][0]), int(anchors__[k][1])), \
    #             (int(anchors__[k][2]), int(anchors__[k][3])), (255,0,0), 1)
    #         cx = ((xyxy[k][0] + xyxy[k][2]) / 2 / 8)
    #         cx2 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 16)
    #         cx3 = ((xyxy[k][0] + xyxy[k][2]) / 2 / 32)
    #         cy = ((xyxy[k][1] + xyxy[k][3]) / 2 / 8)
    #         cy2 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 16)
    #         cy3 = ((xyxy[k][1] + xyxy[k][3]) / 2 / 32)
    #         kk = (cx, cy)
    #         kk2 = (cx2, cy2)
    #         kk3 = (cx3, cy3)
            
    #         acx = ((anchors__[k][0] + anchors__[k][2]) / 2 / 8)
    #         acx2 = ((anchors__[k][0] + anchors__[k][2]) / 2 / 16)
    #         acx3 = ((anchors__[k][0] + xyxy[k][2]) / 2 / 32)
    #         acy = ((anchors__[k][1] + anchors__[k][3]) / 2 / 8)
    #         acy2 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 16)
    #         acy3 = ((anchors__[k][1] + anchors__[k][3]) / 2 / 32)
    #         akk = (acx, acy)
    #         akk2 = (acx2, acy2)
    #         akk3 = (acx3, acy3)
    #         cv2.putText(tmp, str(ch[k]), (int(xyxy[k][0]), int(xyxy[k][1]) + 10), cvfont, 0.5, [255, 0, 0], 1)
    #         cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth_%d.jpg'%k, tmp)
    #     exit(0)
        
    del choose_anchors
    
    index = torch.sort(iou_anchor_truth, descending=True)[1]
    predictind = predictind[index]
    truthind = truthind[index]
    del iou_anchor_truth, index

    keepind = []
    tek = set()
    # predictind = predictind.cpu().numpy()
    predic = list(predictind.cpu().numpy())
    for i in range(len(predic)):
        ind = predic[i]
        if ind not in tek:
            tek.add(ind)
            keepind.append(i)
    pre_keepind = predictind[keepind]
    choose_predict = predicts[pre_keepind]
    choose_masks = masks[pre_keepind]
    choose_label   = labels[truthind[keepind], :]
    
    prenum = 0
    labelnum = 0
    truth_index = truthind[keepind]
    ph, pw = prototype.size()[-1], prototype.size()[-2]
    for i in range(batchsize):
        ch = (pre_keepind >= prenum) & (pre_keepind < (prenum + total_pixel))
        choose_masks = masks[pre_keepind[ch]]

        # cp = predicts[pre_keepind[ch]][:, :2*2]
        # area = cp[:, 2:].prod(1) / (model.imgsize**2)
        # min_xy = (cp[:, 0:2] - cp[:, 2:] / 2.0) * pw / model.imgsize
        # max_xy = (cp[:, 0:2] + cp[:, 2:] / 2.0) * pw / model.imgsize
        # xyxy = torch.concat([min_xy, max_xy], dim = 1)
        
        num = torch.sum(labels[:, 0].long() == i)
        tch = (truth_index >= labelnum) & (truth_index < (labelnum + num))
        choose_la = labels[truth_index[tch]].float()
        area = (choose_la[:, 2*2] - choose_la[:, 2]) * (choose_la[:, 2*2+1] - choose_la[:, 2+1]) / (model.imgsize**2)
        xyxy = choose_la[:, 2:]* pw / model.imgsize

        choose_gtmasks = gtmask[truth_index[tch]].float()
        
        p = prototype[i]
        # pred_mask = torch.matmul(choose_masks, p.view(26+6, -1)).view(-1, p.size()[1], p.size()[-1])
        pred_mask = (choose_masks @ p.view(26+6, -1)).view(-1, p.size()[1], p.size()[-1])
        loss = F.binary_cross_entropy_with_logits(pred_mask, choose_gtmasks, reduction='none')
        # loss = bce2loss(pred_mask, choose_gtmasks)
        ml = (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).mean()
        maskloss += ml
        # maskloss += (loss.mean(dim=(1, 2)) / area).mean()
        prenum += total_pixel
        labelnum += num

    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)

    # #
    indkeep = pre_keepind % total_pixel
    layer_scale = torch.ones(len(pre_keepind), dtype = torch.float32, device = model.device)
    if len(model.yolo)==3:
        kid = indkeep < num_layer[0]
        # k = torch.sum(kid)
        layer_scale[kid] = 1 / torch.sum(kid)
        kidkid = (indkeep < (num_layer[1] + num_layer[0]))
        kl = kidkid & (~kid)
        # kk = torch.sum(kl)
        layer_scale[kl] = 1 / torch.sum(kl)
        kidkid = ~kidkid
        # end = torch.sum(kidkid)
        layer_scale[kidkid] = 1 / torch.sum(kidkid)
        del kid, kl, kidkid, indkeep
    else:
        kid = indkeep < num_layer[0]
        layer_scale[kid] = 1 / torch.sum(kid)
        kid = ~kid
        layer_scale[kid] = 1 / torch.sum(kid)
        del kid, indkeep
    # #

    confidence_predict = predicts[:, (2*2)]
    del predicts
    
    if num_scale:
        count_scale = count_scale[choose_label[:, 1].long()]
        count_scale = torch.clamp(count_scale, 0, 3)

    ciou, diou, iou, giou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    del diou, iou, giou
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))

    iou_loss = (1 - ciou) * layer_scale  # scale * count_scale

    # iou_loss = iou_loss                             # / one_anchor_multilabel
    iouloss = torch.sum(iou_loss)   #/ (len(ciou) / len(model.yolo))
    iounow = torch.mean(ciou)
    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_predict[:, (2*2+1):] # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32, device = model.device) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale  # pos_scale  ciou
    # kk = predicts[confi_masks_rev, (2*2+1):]
    # iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    num_classes = classes.size(1)
    if num_scale:
        count_scale = torch.unsqueeze(count_scale, dim = -1)
        c_l   = torch.mean(bcecls(classes, class_la) * count_scale)
    else:
        # c_l   += torch.mean(torch.pow(classes - class_la, 2))
        layer_scale = layer_scale.unsqueeze(-1)
        assert bce0loss.reduction=='none'
        c_l   = torch.sum(bce0loss(classes, class_la) * (layer_scale / num_classes))  # / (len(classes) / len(model.yolo)) # + bceloss(kk, torch.zeros_like(kk))
        # c_l   = bce0loss(classes, class_la)

    Truth_confid = torch.zeros_like(confidence_predict, dtype = torch.float32, device = model.device)
    Truth_confid[pre_keepind] = ciou.detach().clamp(0).type(Truth_confid.dtype)
    if num_scale:
        pass
        # confi_l   = torch.mean(bcecof(confidence, torch.ones_like(confidence)) * count_scale) + bce1loss(noconf, torch.zeros_like(noconf))
    else:
        # confi_l   += torch.mean(torch.pow(confidence - torch.ones_like(confidence), 2)) +  torch.mean(torch.pow(noconf - torch.zeros_like(noconf), 2))
        # ciou = ciou.unsqueeze(-1)
        # cofobj = torch.ones_like(confidence) * ciou.clamp(0).type(confidence.dtype)
        # confi_l   += bcecof(confidence, cofobj) + bce2loss(noconf, torch.zeros_like(noconf))
        assert bce1loss.reduction=='sum'
        confi_l   = bce1loss(confidence_predict, Truth_confid) / len(confidence_predict)

    hypbox = 0.06 * 3 / len(model.yolo)  # scale to layers
    hypcls = 0.6 * num_classes / 80 * 3 / len(model.yolo)  # scale to classes and layers
    hypobj = 1.0 * (model.imgsize / 640) ** 2 * 3 / len(model.yolo)  # scale to image size and layers

    confi_l *= hypobj
    c_l *= hypcls
    iouloss *= hypbox
    maskloss *= hypbox
    # confi_l *= 1.0
    # c_l *= 0.6
    # iouloss *= 0.06

    cof = torch.mean(confidence_predict[pre_keepind].sigmoid())
    k = confidence_predict.sigmoid()
    ncof = (torch.sum(k) - torch.sum(k[pre_keepind])) / (len(confidence_predict) - len(pre_keepind))
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = (c_l + confi_l + iouloss + maskloss) * batchsize
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, maskloss, 1#len(labels)

# hyp params in yolovfive yolovseven 110  ### no iou_scale  no count_scale    ## iou_scale small anchors not reasonable
def calculate_losses_20230730(prediction, labels, model, count_scale, ignore_thresh, \
                              bce0loss, bce1loss, bce2loss, bcecls, bcecof, mseloss, gtmask, num_scale = False):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.parallel.DataParallel)):
        model = model.module
    predicts = []
    anchors = []
    for i in range(len(model.yolo)):
        anchors.append(prediction[i][1])
        predicts.append(prediction[i][0])
    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    batchsize = prediction[0][0].size()[0]

    del prediction

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    thresh = 0.6 - 0.2 - 0.2
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    n, k, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    iou_anchor_truth = box_iou(labels[:, 2:], anchors)
    pre = 0
    # maxind = torch.tensor([], dtype=torch.long).to(model.device)
    # col_choose = torch.tensor([], dtype=torch.long).to(model.device)
    iou_scale = torch.tensor([], dtype=torch.float).to(model.device)
    confi_masks = torch.tensor([], dtype=torch.bool).to(model.device)
    IOUhigher = torch.zeros((1, 3), dtype = torch.float, device = model.device)

    Tvalue_all, Tindexs_all = torch.sort(iou_anchor_truth, dim = 1, descending=True)
    chnum = len(model.yolo) * 100
    Tvalue_all = Tvalue_all[:, :chnum]
    Tindexs_all = Tindexs_all[:, :chnum]
    del iou_anchor_truth
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)

        Tvalue = Tvalue_all[pre:pre + num, :]
        Tindex = Tindexs_all[pre:pre + num, :]
        confi = torch.zeros(len(anchors), dtype = torch.bool, device = model.device)
        
        iouhigh = torch.zeros((6000, 3), dtype = torch.float, device = model.device)
        pk = 0
        for ia in range(num):
            argind = Tindex[ia]
            argval = Tvalue[ia] 
            mk = argval > thresh
            um = torch.sum(mk)
            if um==0:
                mk[:2] = True
                um=2
            col = argind[mk] + (len(anchors) * i)
            # col_choose = torch.concat([col_choose, col], dim = 0)
            # maxind     = torch.concat([maxind, torch.ones((len(argind[mk])), device = model.device, dtype=torch.long) * ia + pre], dim = 0)
            # iou_scale    = torch.concat([iou_scale, argval[mk]], dim = 0)
            iouhigh[torch.arange(pk, pk + um), 0] = col.float()
            iouhigh[torch.arange(pk, pk + um), 1] = float(ia + pre)
            iouhigh[torch.arange(pk, pk + um), 2] = argval[mk]
            confi[argind[mk]] = True
            pk += um
        iouhigh = iouhigh[:pk, :]
        IOUhigher = torch.concat([IOUhigher, iouhigh], dim = 0)
        confi_masks = torch.concat([confi_masks, confi], dim = 0)
        pre += num

    IOUhigher = IOUhigher[1:]
    index = torch.sort(IOUhigher[:, -1], descending=True)[1]
    IOUhigher = IOUhigher[index]
    IOUhigher = list(IOUhigher.detach().cpu().numpy())
    lastanchor = []
    lasttruth = []
    iou_scale = []
    tek = set()
    for i in range(len(IOUhigher)):
        ind, la, iou = IOUhigher[i]
        if ind not in tek:
            tek.add(ind)
            lastanchor.append(ind)
            lasttruth.append(la)
            iou_scale.append(iou)
    lastanchor = torch.tensor(lastanchor, dtype = torch.long, device = model.device)
    lasttruth = torch.tensor(lasttruth, dtype = torch.long, device = model.device)
    iou_scale = torch.tensor(iou_scale, dtype = torch.float, device = model.device)

    # col_choose = col_choose.detach().cpu().numpy()
    # maxind = maxind.detach().cpu().numpy()
    # for i in range(len(col_choose)):
    #     if col_choose[i] not in tek:
    #         tek.add(col_choose[i])
    #         lastanchor.append(col_choose[i])
    #         lasttruth.append(maxind[i])
    # lastanchor = torch.tensor(lastanchor, dtype = torch.long, device = model.device)
    # lasttruth = torch.tensor(lasttruth, dtype = torch.long, device = model.device)
    
    # col_choose_cpu = lastanchor.detach().cpu().numpy()
    # dic = {}
    # for i in col_choose_cpu:
    #     if i not in dic.keys():
    #         dic[i] = 1
    #     else:
    #         dic[i] += 1
    # one_anchor_multilabel = torch.tensor([dic[i] for i in col_choose_cpu], dtype = torch.float32, device = model.device)
    
    # confi_masks_rev = ((~confi_masks) & delta_k)
    confi_masks_rev = (~confi_masks)
    noconf = predicts[confi_masks_rev, (2*2)].unsqueeze(-1)
    # choose_iou_predict = predicts[col_choose, :]
    # choose_iou_label   = labels[maxind.long(), :]

    choose_cla_predict = predicts[lastanchor, :]
    choose_cla_label   = labels[lasttruth, :]

    del Tvalue_all, Tindexs_all, IOUhigher, confi_masks
    # count_scale = count_scale[choose_label[:, 1].long()] # / 10.0

##########################
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]

    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    for i in torch.unique(lasttruth):
        ch = lasttruth==i
        iounow = iou_scale[ch]
        iouch = iounow / torch.max(iounow)
        # sum = int(torch.sum(ch))
        # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
        iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
##########################
    
    xywh2xyxy(choose_cla_predict[:, 0:(2*2)], model.imgsize, clamp = False)
    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    # ciou, diou, iou, giou = complete_box_iou_no_expand(choose_iou_predict[:, 0:(2*2)], choose_iou_label[:, 2:])
    ciou, diou, iou, giou = complete_box_iou_no_expand(choose_cla_predict[:, 0:(2*2)], choose_cla_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    # w = choose_label[:, 2*2] - choose_label[:, 2]
    # h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    # area = h * w
    # scale =  2.0 - (area / (model.imgsize**2))
    if num_scale:
        iou_loss = (1 - ciou) + (1 - diou) + (1 - iou) + (1 - giou)  * iou_scale
        iou_loss = iou_loss * count_scale
    else:
        iou_loss = (1 - ciou) * iou_scale         # scale * count_scale

    iou_loss = iou_loss # / one_anchor_multilabel
    iouloss += torch.sum(iou_loss)
    iounow = torch.mean(ciou)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    # neg_scale = 0
    classes = choose_cla_predict[:, (2*2+1):]  # * choose_cla_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_cla_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    # kk = predicts[~col_choose, (2*2+1):]
    iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # count_scale = torch.unsqueeze(count_scale, dim = -1)
    assert bcecls.reduction=='sum'
    c_l   += bcecls(classes, class_la * iou_scale)  # * count_scale) # + bceloss(kk, torch.zeros_like(kk))
    # c_l   += bce0loss(classes, class_la) * iou_scale # + bceloss(kk, torch.zeros_like(kk))

    confidence = choose_cla_predict[:, (2*2)].unsqueeze(-1)
    assert bcecof.reduction=='sum' and bce0loss.reduction=='sum'
    confi_l   += bcecof(confidence, torch.ones_like(confidence) * iou_scale) + bce0loss(noconf, torch.zeros_like(noconf))
    # prediou = prediou.unsqueeze(-1)
    # cofobj = torch.ones_like(confidence)  # * prediou.clamp(0).type(confidence.dtype)
    # confi_l   += bce1loss(confidence, cofobj) + bce2loss(noconf, torch.zeros_like(noconf))
    
    # confi_l *= 1.0
    # c_l *= 0.6
    # iouloss *= 0.06

    cof = torch.mean(confidence.sigmoid())
    ncof = torch.mean(noconf.sigmoid())
    cla = torch.mean(classes[torch.arange(len(ll)), ll].sigmoid())
    loss = c_l + confi_l + iouloss
    return loss, c_l, confi_l, iouloss, iounow, cof, ncof, cla, len(labels)

# iou_scale 110 20230729
def calculate_losses_110_20230729(prediction, labels, model, count_scale):
    predicts = []
    anchors = []
    for i in range(len(model.yolo)):
        anchors.append(prediction[i][1])
        predicts.append(prediction[i][0])
    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    batchsize = prediction[0][0].size()[0]
    
    mseloss = torch.nn.MSELoss(reduction='none').to(model.device)
    bceloss = torch.nn.BCELoss().to(model.device)
    bcecls = torch.nn.BCELoss(reduction='none').to(model.device)
    bcecof = torch.nn.BCELoss(reduction='none').to(model.device)

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    thresh = 0.6 - 0.2 - 0.1
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    n, k, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    all_iou = complete_box_iou(anchors, labels[:, 2:])
    pre = 0
    maxind = torch.tensor([], dtype=torch.int).to(model.device)
    col_choose = torch.tensor([], dtype=torch.bool).to(model.device)
    iou_scale = torch.tensor([], dtype=torch.float).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        iou = all_iou[:, pre:pre + num]
        
        T_iou = iou.T
        max_val, max_ind  = torch.max(iou, dim = 1)
        choose = max_val > thresh

        T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
        max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
        # choose[...] = False
        choose[T_max_ind] = True
        
        while len(max_ind[choose]) == 0:
            thresh = np.exp(-thresh) * thresh
            max_val, max_ind  = torch.max(iou, dim = 1)
            choose = max_val > thresh
            T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
            max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
            choose[T_max_ind] = True
        
        maxind = torch.concat([maxind, max_ind[choose] + pre], dim = 0)
        col_choose = torch.concat([col_choose, choose], dim = 0)
        iou_column = iou[choose]
        iou_ch = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        iou_scale = torch.concat([iou_scale, iou_ch], dim = 0)
        
        pre += num

    choose_predict = predicts[col_choose, :]
    choose_label   = labels[maxind, :]
    
    # count_scale = count_scale[choose_label[:, 1].long()] / 10.0
    
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]
    
    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    for i in torch.unique(maxind):
        ch = maxind==i
        iounow = iou_scale[ch]
        iouch = iounow / torch.max(iounow)
        # sum = int(torch.sum(ch))
        # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
        iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
    
    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)
    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    prediou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    w = choose_label[:, 2*2] - choose_label[:, 2]
    h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    area = h * w
    scale =  2.0 - (area / (model.imgsize**2))
    iou_loss = (1 - prediou) * scale  * iou_scale # * count_scale
    iouloss += torch.mean(iou_loss)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    neg_scale = 0
    classes = choose_predict[:, (2*2+1):]  #* choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    kk = predicts[~col_choose, (2*2+1):]
    iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # count_scale = torch.unsqueeze(count_scale, dim = -1)
    c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # * count_scale) # + bceloss(kk, torch.zeros_like(kk))

    confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    noconf = predicts[~col_choose, (2*2)].unsqueeze(-1)
    # confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * iou_scale * count_scale) + bceloss(1 - noconf, torch.ones_like(noconf))
    confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * iou_scale) + bceloss(1 - noconf, torch.ones_like(noconf))

    confi_l *= 0.7
    c_l *= 0.3
    iouloss *= 0.06
    loss = ( c_l + confi_l + iouloss ) * batchsize
    return loss, mse, c_l, confi_l, iouloss

def calculate_losses_20230728(prediction, labels, model, count_scale):
    predicts = []
    anchors = []
    for i in range(len(model.yolo)):
        anchors.append(prediction[i][1])
        predicts.append(prediction[i][0])
    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    batchsize = prediction[0][0].size()[0]
    
    mseloss = torch.nn.MSELoss(reduction='none').to(model.device)
    bceloss = torch.nn.BCELoss().to(model.device)
    bcecls = torch.nn.BCELoss(reduction='none').to(model.device)
    bcecof = torch.nn.BCELoss(reduction='none').to(model.device)

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    thresh = 0.6 - 0.2 - 0.1
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    n, k, kl = predicts.size()
    predicts = torch.reshape(predicts, (-1, kl))

    all_iou = complete_box_iou(anchors, labels[:, 2:])
    pre = 0
    maxind = torch.tensor([], dtype=torch.int).to(model.device)
    col_choose = torch.tensor([], dtype=torch.bool).to(model.device)
    iou_scale = torch.tensor([], dtype=torch.float).to(model.device)
    for i in range(batchsize):
        num = torch.sum(labels[:, 0].long() == i)
        iou = all_iou[:, pre:pre + num]
        
        T_iou = iou.T
        max_val, max_ind  = torch.max(iou, dim = 1)
        choose = max_val > thresh

        T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
        max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
        # choose[...] = False
        choose[T_max_ind] = True
        
        while len(max_ind[choose]) == 0:
            thresh = np.exp(-thresh) * thresh
            max_val, max_ind  = torch.max(iou, dim = 1)
            choose = max_val > thresh
            T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
            max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
            choose[T_max_ind] = True
        
        maxind = torch.concat([maxind, max_ind[choose] + pre], dim = 0)
        col_choose = torch.concat([col_choose, choose], dim = 0)
        iou_column = iou[choose]
        iou_ch = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        iou_scale = torch.concat([iou_scale, iou_ch], dim = 0)
        
        pre += num

    choose_predict = predicts[col_choose, :]
    choose_label   = labels[maxind, :]
    
    # count_scale = count_scale[choose_label[:, 1].long()] / 10.0
    
    # index = np.lexsort((iou_scale.cpu().numpy(), maxind.cpu().numpy()))
    # choose_predict = choose_predict[index]
    # choose_label = choose_label[index]
    # iou_scale = iou_scale[index]
    # maxind = maxind[index]
    
    # p_re = maxind[0]
    # ind = 0
    # kk = iou_scale.clone()
    # for i in range(len(maxind) + 1):
    #     if i == len(maxind) or p_re != maxind[i]:
    #         kk[ind:i] = kk[ind:i] / kk[i - 1]
    #         if i != len(maxind):
    #             p_re = maxind[i]
    #             ind = i
    
    # 20230729
    # for i in torch.unique(maxind):
    #     ch = maxind==i
    #     iounow = iou_scale[ch]
    #     iouch = iounow / torch.max(iounow)
    #     # sum = int(torch.sum(ch))
    #     # tmp = torch.linspace(1.0, np.exp(-sum), sum, )
    #     iou_scale[ch] = iouch

    # kkk = torch.sum(kk!=iou_scale)
    
    xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)
    # indexe = torch.arange(len(choose_label))
    # prediou = complete_box_iou(choose_predict[:, 0:(2*2)], choose_label)
    # prediou = prediou[indexe, indexe]
    prediou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label[:, 2:])
    # kkk = torch.sum(prediou_!=prediou)
    w = choose_label[:, 2*2] - choose_label[:, 2]
    h = choose_label[:, 2*2+1] - choose_label[:, 2+1]
    area = h * w
    scale =  2.0 - (area / (model.imgsize**2))
    iou_loss = (1 - prediou) * scale  * iou_scale * count_scale
    iouloss += torch.mean(iou_loss)

    # mse  += torch.mean(mseloss(choose_predict[:, 0:(2*2)]/model.imgsize, choose_label/model.imgsize)  * iou_scale)
    # https://github.com/WongKinYiu/yolov7/blob/main/utils/loss.py
    # pos_scale = 1 - (1/60.0)      # 1
    # neg_scale = 1/60.0            # 0
    pos_scale = 1
    neg_scale = 0
    classes = choose_predict[:, (2*2+1):] # * choose_predict[:, 2*2].unsqueeze(-1)
    class_la = torch.zeros_like(classes, dtype = torch.float32) # * neg_scale
    ll  = choose_label[:, 1].long()
    class_la[torch.arange(len(ll)), ll] = pos_scale
    kk = predicts[~col_choose, (2*2+1):]
    iou_scale = torch.unsqueeze(iou_scale, dim = -1)
    # count_scale = torch.unsqueeze(count_scale, dim = -1)
    # c_l   += torch.mean(bcecls(classes, class_la) * (1 + iou_scale)) + bceloss(kk, torch.zeros_like(kk))
    c_l   += torch.mean(bcecls(classes, class_la) * iou_scale) # + bceloss(kk, torch.zeros_like(kk))

    confidence = choose_predict[:, (2*2)].unsqueeze(-1)
    noconf = predicts[~col_choose, (2*2)].unsqueeze(-1)
    confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence)) * iou_scale) + bceloss(1 - noconf, torch.ones_like(noconf))

    mse /= batchsize
    c_l /= batchsize
    confi_l /= batchsize
    iouloss /= batchsize

    confi_l *= 0.7
    c_l *= 0.3
    iouloss *= 0.06
    loss = ( c_l + confi_l + iouloss ) * batchsize
    return loss, mse, c_l, confi_l, iouloss

def calculate_parallel_losses(prediction, labels, device, model):
    predicts = []
    anchors = []
    for i in range(len(model.yolo)):
        anchors.append(prediction[i][1])
        predicts.append(prediction[i][0])
    anchors = torch.concat(anchors, dim=(0)).float()
    labels  = labels.float()
    predicts = torch.concat(predicts, dim=(1))
    batchsize = prediction[0][0].size()[0]
    
    mseloss = torch.nn.MSELoss().to(model.device)
    bceloss = torch.nn.BCELoss().to(model.device)
    bcecls = torch.nn.BCELoss(reduction='none').to(model.device)
    bcecof = torch.nn.BCELoss(reduction='none').to(model.device)

    loss = torch.tensor(0, dtype=torch.float32).to(model.device)
    mse = torch.tensor(0, dtype=torch.float32).to(model.device)
    c_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    confi_l = torch.tensor(0, dtype=torch.float32).to(model.device)
    thresh = 0.6 - 0.2 - 0.1
    iouloss = torch.tensor(0, dtype=torch.float32).to(model.device)

    labels[:, 2:] = labels[:, 2:] * model.imgsize
    xywh2xyxy(labels[:, 2:], model.imgsize)
    iou_col = complete_box_iou(anchors, labels[:, 2:])
    pre = 0
    for i in range(batchsize):
        pre = predicts[i, ...]
        index = labels[:, 0].long() == i
        num = torch.sum(index)
        la = labels[index].clone()
        xyxy = labels[index, 2:].clone()

        # tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * 2**(2**3)
        # for k in range(len(anchors)):
        #     if (k+1)%3==0:
        #         cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/%s.jpg'%str(k//3), tmp)
        #         tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * 2**(2**3)
        #     cv2.rectangle(tmp, (int(anchors[k][0]), int(anchors[k][1])), \
        #         (int(anchors[k][2]), int(anchors[k][3])), \
        #         [np.random.randint(255),np.random.randint(255),np.random.randint(255)], 2)
        #     # if k == 100:
        #     #     break
        # tmp = np.ones((model.imgsize, model.imgsize), dtype=np.uint8) * 2**(2**3)
        # for k in range(len(xyxy)):
        #     cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
        #         (int(xyxy[k][2]), int(xyxy[k][3])), (0,0,255), 2)
        # cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth.jpg', tmp)

        # iou = box_iou(anchors, xyxy)
        # iou = complete_box_iou(anchors, xyxy)
        
        # T_iou = iou.T
        # max_val, max_ind  = torch.max(iou, dim = 1)
        # choose = max_val > thresh

        # T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
        # max_ind[T_max_ind] = torch.arange(0, len(xyxy))
        # choose[T_max_ind] = True

        # for i in range(len(xyxy)):
        #     ch = T_iou[i, :] > thresh - 0.2
        #     max_ind[ch] = i
        #     choose[ch] = True
        
        # anchors = anchors[choose]
        # xyxy    = xyxy[max_ind[choose]]
        # for k in range(len(anchors)):
        #     tmp = np.ones((model.imgsize, model.imgsize, 3), dtype=np.uint8) * 2**(2**3)
        #     cv2.rectangle(tmp, (int(xyxy[k][0]), int(xyxy[k][1])), \
        #         (int(xyxy[k][2]), int(xyxy[k][3])), (0,0,255), 2)
        #     cv2.rectangle(tmp, (int(anchors[k][0]), int(anchors[k][1])), \
        #         (int(anchors[k][2]), int(anchors[k][3])), (255,0,0), 1)
        #     cv2.imwrite(r'/root/project/Pytorch_YOLOV3/datas/imshow/truth_%d.jpg'%k, tmp)
        # exit(0)

        # iou = box_iou(anchors, xyxy)
        
        iou = iou_col[:, pre:pre + num]
        T_iou = iou.T
        max_val, max_ind  = torch.max(iou, dim = 1)
        choose = max_val > thresh

        T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
        max_ind[T_max_ind] = torch.arange(0, num).to(model.device)
        # choose[...] = False
        choose[T_max_ind] = True

        # for i in range(len(xyxy)):
        #     ch = T_iou[i, :] > thresh - 0.2
        #     max_ind[ch] = i
        #     choose[ch] = True
        
        choose_predict = pre[choose, :]
        choose_label   = xyxy[max_ind[choose], :]
        choose_l   = la[max_ind[choose], :]
        iou_column = iou[choose]
        iou_scale   = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        while len(choose_label)==0:
            thresh = np.exp(-thresh) * thresh
            choose = max_val > thresh
            T_max_val, T_max_ind  = torch.max(T_iou, dim = 1)
            max_ind[T_max_ind] = torch.arange(0, num)
            choose[T_max_ind] = True
            choose_predict = pre[choose, :]
            choose_label   = xyxy[max_ind[choose], :]
            choose_l   = la[max_ind[choose], :]

            iou_column = iou[choose]
            iou_scale = iou_column[torch.arange(len(iou_column)), max_ind[choose]]
        
        pre += num

        xywh2xyxy(choose_predict[:, 0:(2*2)], model.imgsize, clamp = False)
        # indexe = torch.arange(len(cpxy))
        # prediou_ = complete_box_iou(cpxy, choose_label)
        # prediou_ = prediou_[indexe, indexe]
        prediou = complete_box_iou_no_expand(choose_predict[:, 0:(2*2)], choose_label)
        # kkk = torch.sum(prediou_!=prediou)
        h = choose_l[:, -1]
        w = choose_l[:, -2]
        area = h * w
        scale =  2.0 - (area / (model.imgsize**2))
        iou_loss = (1 - prediou) * scale
        iouloss += torch.mean(iou_loss)

        # mse  += mseloss(cpxy/model.imgsize, choose_label/model.imgsize)

        pos_scale = 1                  # 1
        neg_scale = 0                     # 0
        classes = choose_predict[:, (2*2+1):] # * choose_predict[:, 2*2].unsqueeze(-1)
        class_la = torch.ones_like(classes) * neg_scale
        ll  = choose_l[:, 1].long()
        class_la[torch.arange(len(ll)), ll] = pos_scale
        kk = pre[~choose, (2*2+1):]
        iou_scale = torch.unsqueeze(iou_scale, -1)
        c_l   += torch.mean(bcecls(classes, class_la)) + bceloss(kk, torch.zeros_like(kk))

        confidence = choose_predict[:, (2*2)].unsqueeze(-1)
        noconf = pre[~choose, (2*2)].unsqueeze(-1)
        confi_l   += torch.mean(bcecof(confidence, torch.ones_like(confidence))) + bceloss(noconf, torch.zeros_like(noconf))
    
    mse /= batchsize
    c_l /= batchsize
    confi_l /= batchsize
    iouloss /= batchsize

    loss = ( c_l + confi_l + iouloss ) * batchsize
    return loss, mse, c_l, confi_l, iouloss
