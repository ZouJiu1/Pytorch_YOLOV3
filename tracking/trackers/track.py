# Ultralytics YOLO 🚀, AGPL-3.0 license

from functools import partial

import torch

from .bot_sort import BOTSORT
from .byte_tracker import BYTETracker

TRACKER_MAP = {'bytetrack': BYTETracker, 'botsort': BOTSORT}

class database:
    def __init__(self, bs) -> None:
        self.bs = bs

class ALL_box:
    def __init__(self, boxes, orishape) -> None:
        self.conf = []
        self.xyxy = []
        self.cls = []
        self.boxes = boxes
        self.orishape = orishape

    def __getitem__(self, id):
        return self.__class__(boxes = self.boxes[id], orishape = self.orishape)

    def __len__(self):
        return len(self.conf)

class ALL_result:
    def __init__(self, orishape=None) -> None:
        self.boxes = []
        self.orishape = orishape
    
    def add(self, boxes):
        self.boxes = boxes

    def __getitem__(self, id):
        r = self.__class__(self.orishape)
        boxes = getattr(self, 'boxes')
        setattr(r, 'boxes', boxes[id])
        return r

    def clipbox(self, boxes):
        orishape = self.orishape
        if isinstance(boxes, torch.Tensor):
            boxes[..., 0] = boxes[..., 0].clamp_(0, orishape[1])
            boxes[..., 1] = boxes[..., 1].clamp_(0, orishape[0])
            boxes[..., 2] = boxes[..., 2].clamp_(0, orishape[1])
            boxes[..., 3] = boxes[..., 3].clamp_(0, orishape[0])
        else:
            boxes[..., [0, 2]] = boxes[..., [0,2]].clip(0, orishape[1])
            boxes[..., [1, 3]] = boxes[..., [1,3]].clip(0, orishape[0])
    
    def update(self, boxes):
        self.clipbox(boxes)
        self.boxes = ALL_box(boxes, self.orishape)
        
class ALL_Predictor:
    def __init__(self, bs=1) -> None:
        self.trackers = []
        self.bs = bs
        self.results = [ALL_result() for i in range(bs)]
        self.dataset = database(bs)
        self.batch = []
    
    def batch_add(self, img):
        self.batch.append(img)

botcfg = {'tracker_type':'botsort', 'track_high_thresh':'0.5', 'track_low_thresh': '0.1', 'new_track_thresh':'0.6', \
              'track_buffer':'30', 'match_thresh':'0.8', 'cmc_method':'sparseOptFlow', 'proximity_thresh':'0.5', \
              'appearance_thresh':'0.25', 'with_reid':False}
              
bytecfg = {'tracker_type':'bytetrack', 'track_high_thresh':'0.5', 'track_low_thresh': '0.1', 'new_track_thresh':'0.6', \
              'track_buffer':'30', 'match_thresh':'0.8'}

class cfgcreate:
    def __init__(self, cfg) -> None:
        for key, value in cfg.items():
            try:
                if not isinstance(value, bool):
                    value = float(value)
            except:
                pass
            setattr(self, key, value)
        
def on_predict_start(the_batchsize):
    global botcfg, bytecfg
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
    Raises:
        AssertionError: If the tracker_type is not 'bytetrack' or 'botsort'.
    """
    cfg = cfgcreate(botcfg)
    assert cfg.tracker_type in ['bytetrack', 'botsort'], \
        f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    p = ALL_Predictor()
    for _ in range(the_batchsize):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    p.trackers = trackers
    return p

def on_predict_postprocess_end(predictor: ALL_Predictor):
    """Postprocess detected boxes and update with object tracking."""
    bs = predictor.dataset.bs
    im0s = predictor.batch
    for i in range(bs):
        det = predictor.results[i].boxes
        if len(det) == 0:
            continue
        tracks = predictor.trackers[i].update(det, im0s[i])
        if len(tracks) == 0:
            continue
        idx = tracks[:, -1].astype(int)
        predictor.results[i] = predictor.results[i][idx]
        predictor.results[i].update(boxes=torch.as_tensor(tracks[:, :-1]))
