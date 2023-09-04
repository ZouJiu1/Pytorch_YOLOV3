## Pytorch_YOLOV3 from COCO scratch <br>

### upgrade version 7 2023-08-09
I found why the training before is slowly and can not merge finally, it is very simple to understand.

I used the **torch.mean** to calculate the loss of non_confidence before, so it loss decrease very slowly and the derivation or the gradient backpropagation is slowly, the non_confidence gradient backpropagation will multiply the $\frac{1}{number~of~non\_confidence}$, the gradient will be very very small, the non_confidence loss work for nothing. using torch.mean need so many epoch to training and nonconf can not be trained, the real training epoch is little. so the negative confidence or non confidence can not be trained. the recall is low and precision is low.

but the training codes and model is correct.

# Detection model

[model_e30_map https://www.aliyundrive.com/s/xVYNAiaa19c 提取码: 16bn](https://www.aliyundrive.com/s/xVYNAiaa19c)

[model_e51_map https://www.aliyundrive.com/s/F7wRTcitbkB 提取码: gn85](https://www.aliyundrive.com/s/F7wRTcitbkB)

**Train**
```
python allcodes/train_yolovkkn.py
python allcodes/train_yolovkkn_distribute.py
```

**Predict**
```
python allcodes/predict_yolovkkn.py
```

##### Things different

I write several different loss calculation function with torch.sum instead of mean and use yolov3-tiny network to training, it is training in gpu server with multi gpu cards, it needs much money for renting it, so I train very little epoch and use small batchsize to saving money without adjust hyper_params. The training result is very good compared with before.

those references are [darknet_yolov3 https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), [darknet https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet), [yolov* https://github.com/ultralytics](https://github.com/ultralytics). Considering the reference, those loss functions are:

the **calculate_losses_darknet** training is not stable so using the ciou loss to replace the coordinates loss for now. it is the **calculate_losses_darknetRevise**, the correspondding web is https://github.com/pjreddie/darknet.

the **calculate_losses_Alexeydarknet** is corresponding to https://github.com/pjreddie/darknet.

the **calculate_losses_yolofive** is corresponding to https://github.com/ultralytics

the **calculate_losses_20230730** is writed by myself, it use iou to choose anchor and calculate the loss. I add the num_scale to balance different categories, and the iou scale to balance the difference anchor or prediction loss.

you can find those files in the models directory.

```
calculate_losses_darknetRevise(...)
calculate_losses_darknet(...)
calculate_losses_Alexeydarknet(...)
...
calculate_losses_yolofive(...)
calculate_losses_20230730(...)
...
```
the training result with yolov3-tiny alexeydarknet, 70 epoch, train from scratch without pretrained model and without data augmentation

**the training result with yolov5s, 7 epoch mAP 16%,  10 epoch 21.2% in val2017, train from scratch without pretrained model and without data augmentation** 

<img src='.\images\0 .jpg' width = 100%/>
<img src='.\images\1.jpg' width = 100%/>
<img src='.\images\-1.jpg' width = 100%/>
<img src='.\images\2.jpg' width = 100%/>
<img src='.\images\3.jpg' width = 100%/>
<img src='.\images\4.jpg' width = 100%/>
<img src='.\images\5.jpg' width = 100%/>
<img src='.\images\6.jpg' width = 100%/>
<img src='.\images\7.jpg' width = 100%/>
<img src='.\images\8.jpg' width = 100%/>
<img src='.\images\9.jpg' width = 100%/>
<img src='.\images\10.jpg' width = 100%/>
<img src='.\images\11.jpg' width = 100%/>
<img src='.\images\12.jpg' width = 100%/>
<img src='.\images\13.jpg' width = 100%/>

### dataset

the training datast is coco, which website is [https://cocodataset.org/#download](https://cocodataset.org/#download)

## Tracking
the tracking result is here: waiting for minutes

<img src="./gif.gif" width = "100%" />
<img src="./gifman.gif" width = "100%" />

### run command
**Tracking**
```
python tracking/tracking.py
```

### tracking dataset download
[MVI_39031.zip     https://www.aliyundrive.com/s/PtnZBBf2E2V       提取码: e3q7](https://www.aliyundrive.com/s/PtnZBBf2E2V)

[MOT16-06-raw.webm https://www.aliyundrive.com/s/tUwZH1H5gET       提取码: p50u](https://www.aliyundrive.com/s/tUwZH1H5gET)

# Model Segment

**Train**
```
python allcodes/train_yolovkkn_seg.py
python allcodes/train_yolovkkn_seg_distribute.py
```

**Predict**
```
python allcodes/predict_yolovkkn_seg.py
```
**Model**
model_e60seg_map[0.485051__0.295817]_lnan_2023-09-03.pt [https://www.aliyundrive.com/s/bLnC733gX5C 提取 nu63  ](https://www.aliyundrive.com/s/bLnC733gX5C)   

**Result**
<img src='.\images\segshow\000000004495.jpg' width = 33%/><img src='.\images\segshow\000000025424.jpg' width = 33%/><img src='.\images\segshow\000000033114.jpg' width = 33%/><br>
<img src='.\images\segshow\000000089556.jpg' width = 33%/><img src='.\images\segshow\000000112798.jpg' width = 33%/><img src='.\images\segshow\000000122962.jpg' width = 33%/><br>
<img src='.\images\segshow\000000133567.jpg' width = 33%/><img src='.\images\segshow\000000181542.jpg' width = 33%/><img src='.\images\segshow\000000205282.jpg' width = 33%/><br>
<img src='.\images\segshow\000000228436.jpg' width = 33%/><img src='.\images\segshow\000000268375.jpg' width = 33%/><img src='.\images\segshow\000000270908.jpg' width = 33%/><br>
<img src='.\images\segshow\000000297147.jpg' width = 33%/><img src='.\images\segshow\000000344816.jpg' width = 33%/><img src='.\images\segshow\000000385719.jpg' width = 33%/><br>
<img src='.\images\segshow\000000504635.jpg' width = 33%/><img src='.\images\segshow\000000546829.jpg' width = 33%/><img src='.\images\segshow\000000568213.jpg' width = 33%/><br>


# Model Segment＋Pose Keypoint  Person

**Train**
```
python allcodes/train_yolovkkn_seg_keypoint.py
python allcodes/train_yolovkkn_seg_keypoint_distribute.py
```

**Predict**
```
python allcodes/predict_yolovkkn_seg_keypoint.py
```
**Model**
model_e39segkpt_map[0.502372__0.006359]_l155.764_2023-09-03.pt [https://www.aliyundrive.com/s/bLnC733gX5C 提取 nu63  ](https://www.aliyundrive.com/s/bLnC733gX5C)   

**Result**

the result including segment and person pose keypoint

<img src='.\images\keypoint_seg\000000001000.jpg' width = 33%/><img src='.\images\keypoint_seg\000000031093.jpg' width = 33%/><img src='.\images\keypoint_seg\000000041990.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000052591.jpg' width = 33%/><img src='.\images\keypoint_seg\000000060347.jpg' width = 33%/><img src='.\images\keypoint_seg\000000072281.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000086956.jpg' width = 33%/><img src='.\images\keypoint_seg\000000091500.jpg' width = 33%/><img src='.\images\keypoint_seg\000000118921.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000172877.jpg' width = 33%/><img src='.\images\keypoint_seg\000000173830.jpg' width = 33%/><img src='.\images\keypoint_seg\000000181542.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000186449.jpg' width = 33%/><img src='.\images\keypoint_seg\000000257478.jpg' width = 33%/><img src='.\images\keypoint_seg\000000274411.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000295478.jpg' width = 33%/><img src='.\images\keypoint_seg\000000302536.jpg' width = 33%/><img src='.\images\keypoint_seg\000000309964.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000345466.jpg' width = 33%/><img src='.\images\keypoint_seg\000000357737.jpg' width = 33%/><img src='.\images\keypoint_seg\000000391722.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000398028.jpg' width = 33%/><img src='.\images\keypoint_seg\000000410496.jpg' width = 33%/><img src='.\images\keypoint_seg\000000412362.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000463174.jpg' width = 33%/><img src='.\images\keypoint_seg\000000470173.jpg' width = 33%/><img src='.\images\keypoint_seg\000000477288.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000498807.jpg' width = 33%/><img src='.\images\keypoint_seg\000000500478.jpg' width = 33%/><img src='.\images\keypoint_seg\000000515579.jpg' width = 33%/><br>
<img src='.\images\keypoint_seg\000000517056.jpg' width = 33%/><img src='.\images\keypoint_seg\000000554579.jpg' width = 33%/><img src='.\images\keypoint_seg\000000574823.jpg' width = 33%/><br>
-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------
### upgrade version 6 2022-11-13 

Maybe I know the reason why the mAP of validation dataset is so slow in version3 2022-09

the first reason is that I used a big network yolov3, yolov3 is big, it has so many parameters and the training dataset is little just thousand, and I didn't used a pretrained model, and the train epoch <300. So it must be overfitted and not merged

another reason which is I used albumentations to augment images which old version is lower than 1.2.0 and the labels used in training and the labels used in validation is different. now I find it and upgrade the albumentations to version 1.3.0. 

other reasons are the codes level ...

##### The solution
I don't have a computer which has a gpu,  I rent a computer which has gpu before.  I use [a small network](https://github.com/dog-qiuqiu/Yolo-Fastest) using pytorch from other person, which name is yolofastest and you can visit it from [thisweb](https://github.com/dog-qiuqiu/Yolo-Fastest), yolofastest use the darknet to train.

##### The training result
this time, the network is merged and the mAP is higher than before, you can train your own dataset using command
use `allcodes\\preprocess.py or allcodes\\preprocess_v.py` get data<br>
`
python allcodes/train_yolofastest.py
`

### --------------==================version 3 2022-09========------------------------------------ <br>
### download voc data 
[https://pjreddie.com/projects/pascal-voc-dataset-mirror/](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)

then use `allcodes\\preprocess.py or allcodes\\preprocess_v.py` get data<br>
or you can download it from here [https://pan.baidu.com/s/1nzUiAwUWD8J_lV7qkTdq4w?pwd=u2dn](https://pan.baidu.com/s/1nzUiAwUWD8J_lV7qkTdq4w?pwd=u2dn)
### models
[https://pan.baidu.com/s/1lemOGC5zwJIcOcxWumCcRQ?pwd=9r22](https://pan.baidu.com/s/1lemOGC5zwJIcOcxWumCcRQ?pwd=9r22)
### use 3362 images with 2 classes person and car to train the model from scratch. the model is overfit, so the validation images is not so good.
```
#run
python allcodes/train_730.py
#predict
python allcodes/predict_730.py
```
#### train_overfit
#### validation
#### you can download yolov3 weight from here [https://pan.baidu.com/s/1hWIxV2MggrzL_vlnbMbP_w?pwd=sugw](https://pan.baidu.com/s/1hWIxV2MggrzL_vlnbMbP_w?pwd=sugw)

### --------------==================version 2 2022-09========------------------------------------ <br>
data is same as version 3
### models
[https://pan.baidu.com/s/16FPm_aOxI3hoJoQ0exAmVQ?pwd=xj5k](https://pan.baidu.com/s/16FPm_aOxI3hoJoQ0exAmVQ?pwd=xj5k)

#### train is not very good, maybe it is the yolohead coordinates problem

```
#run
python allcodes/train_717_730.py
#predict
python allcodes/predict_717_730.py
```

### --------------==================version 1 2021-08========------------------------------------ <br>
##### dataset used is voc2007, processed dataset can download from [processedvoc20072012](https://share.weiyun.com/NLjLT13V), unzip it to datas directory，[voc pretrained model](https://share.weiyun.com/7sTyVd7N)

```
#run
python models/train.py
#predict
python predict.py
```

## Reference
[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)<br>
[https://github.com/Peterisfar/YOLOV3](https://github.com/Peterisfar/YOLOV3)<br>
[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)<br>
[https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)<br>
[https://github.com/ultralytics](https://github.com/ultralytics)<br>
[https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)<br>
[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)<br>
[https://github.com/mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)<br>
[https://github.com/Ray-Luo/YOLOV3-PyTorch](https://github.com/Ray-Luo/YOLOV3-PyTorch)<br>
[https://github.com/DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)<br>
