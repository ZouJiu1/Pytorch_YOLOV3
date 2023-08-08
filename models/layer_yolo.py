import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor

def xywh2xyxy(inputs, imgsize, clamp=True):
    min_xy = inputs[:, 0:2] - inputs[:, 2:] / 2.0
    max_xy = inputs[:, 0:2] + inputs[:, 2:] / 2.0
    inputs[:, 0:2] = min_xy
    inputs[:, 2:]  = max_xy
    # if clamp:
    #     inputs = torch.clamp(inputs, 0, imgsize-1)
    return inputs

class yololayer(nn.Module):
    def __init__(self, device, num_anchors, num_classes):
        super(yololayer, self).__init__()
        self.device = device
        self.grid = torch.tensor([[],[]])
        self.no = num_classes + 5
        self.num_anchors = num_anchors
        self.result = {}
        # self.sig = nn.Sigmoid()

    def forward(self, prediction, anchors, imgsize):
        self.stride = imgsize//prediction.size(2)
        batch_size, _, height, width = prediction.size() #batch_size, (5+num_classes)*3, width, height

        #prediction [2, 75, 13, 13]
        # prediction = prediction.view((batch_size, self.num_anchors, height, width, self.no)) #[2, 3, 13, 13, 25]
        prediction = prediction.view(batch_size, self.num_anchors, self.no, height, width).permute(0, 1, 3, 4, 2).contiguous()

        if height not in self.result.keys():
            self.grid__ = self._make_grid(height, width).to(self.device)
            self.grid__ = torch.repeat_interleave(self.grid__, self.num_anchors, dim=1) * self.stride
            self.wh = torch.ones_like(self.grid__) * anchors
            result = torch.concat([self.grid__, self.wh], axis = -1)
            result = result.view(-1, 2*2)
            result = xywh2xyxy(result, imgsize, clamp=False)
            self.result[height] = result
        # # assert 1==0, self._make_grid(prediction.size()[2], prediction.size()[2]).size()
        # if not self.training:
            # print(prediction.size(), self.grid[0].size())
        if self.grid[0].size()[0] != prediction.size()[2]:
            self.grid = self._make_grid(prediction.size()[2], prediction.size()[2]).to(self.device)

        # yololayer_largearea   calculate_losses_darknet
        prediction[..., 0:2] = (prediction[..., 0:2].sigmoid() + self.grid) * self.stride #x #[2, 3, 13, 13]
        prediction[..., 2:4] = torch.exp(prediction[..., 2:4]) * anchors #wh #[2, 3, 13, 13, 2]
        prediction[..., 4:] = prediction[..., 4:].sigmoid()
        
        # calculate_losses_yolofive
        # prediction[..., 0:2] = (prediction[..., 0:2].sigmoid() * 2 - 0.6 + 0.1 + self.grid) * self.stride #x #[2, 3, 13, 13]
        # prediction[..., 2:4] = torch.square(prediction[..., 2:4].sigmoid()) * 2 * 2 * anchors #wh #[2, 3, 13, 13, 2]
        # prediction[..., 4:] = prediction[..., 4:].sigmoid()
        
        prediction = prediction.view(batch_size, -1, self.no)

        return prediction, self.result[height]

    def _make_grid(self, nx, ny):
        # x_coord = torch.arange(width).repeat(height, 1).to(self.device)    #[13, 13]
        # y_coord = torch.transpose((torch.arange(height).repeat(width, 1)), 0, 1).to(self.device)     #[13, 13]
        # return [x_coord, y_coord]
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


def _upcast(t: Tensor) -> Tensor:
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

def box_area(boxes: Tensor) -> Tensor:
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
def _box_inter_union(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


# Implementation adapted from https://github.com/facebookresearch/detr/blob/master/util/box_ops.py
def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return generalized intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise generalized IoU values
        for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union

    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    areai = whi[:, :, 0] * whi[:, :, 1]

    return iou - (areai - union) / areai


def complete_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return complete intersection-over-union (Jaccard index) between two sets of boxes.
    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7
    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise complete IoU values
        for every element in boxes1 and boxes2
    """
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)

    diou, iou = _box_diou_iou(boxes1, boxes2, eps)

    w_pred = boxes1[:, None, 2] - boxes1[:, None, 0]
    h_pred = boxes1[:, None, 3] - boxes1[:, None, 1]

    w_gt = boxes2[:, 2] - boxes2[:, 0]
    h_gt = boxes2[:, 3] - boxes2[:, 1]

    v = (4 / (torch.pi**2)) * torch.pow(torch.atan(w_pred / h_pred) - torch.atan(w_gt / h_gt), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return diou - alpha * v


def distance_box_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tensor:
    """
    Return distance intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes
        eps (float, optional): small number to prevent division by zero. Default: 1e-7

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise distance IoU values
        for every element in boxes1 and boxes2
    """
    boxes1 = _upcast(boxes1)
    boxes2 = _upcast(boxes2)
    diou, _ = _box_diou_iou(boxes1, boxes2, eps=eps)
    return diou


def _box_diou_iou(boxes1: Tensor, boxes2: Tensor, eps: float = 1e-7) -> Tuple[Tensor, Tensor]:

    iou = box_iou(boxes1, boxes2)
    lti = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rbi = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    whi = _upcast(rbi - lti).clamp(min=0)  # [N,M,2]
    diagonal_distance_squared = (whi[:, :, 0] ** 2) + (whi[:, :, 1] ** 2) + eps
    # centers of boxes
    x_p = (boxes1[:, 0] + boxes1[:, 2]) / 2
    y_p = (boxes1[:, 1] + boxes1[:, 3]) / 2
    x_g = (boxes2[:, 0] + boxes2[:, 2]) / 2
    y_g = (boxes2[:, 1] + boxes2[:, 3]) / 2
    # The distance between boxes' centers squared.
    centers_distance_squared = (_upcast((x_p[:, None] - x_g[None, :])) ** 2) + (
        _upcast((y_p[:, None] - y_g[None, :])) ** 2
    )
    # The distance IoU is the IoU penalized by a normalized
    # distance between boxes' centers squared.
    return iou - (centers_distance_squared / diagonal_distance_squared), iou