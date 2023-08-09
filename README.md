## Pytorch_YOLOV3 from COCO scratch <br>

### upgrade version 7 2023-08-09
I found why the training before is slowly and can not merge finally, it is very simple to understand.

I used the **torch.mean** to calculate the loss of non_confidence before, so it loss decrease very slowly and the derivation or the gradient backpropagation is slowly, the non_confidence gradient backpropagation will multiply the $\frac{1}{number~of~non\_confidence}$, the gradient will be very very small, the non_confidence loss work for nothing. using torch.mean need so many epoch to training and nonconf can not be trained, the real training epoch is little. so the negative confidence or non confidence can not be trained. the recall is low and precision is low.

but the training codes and model is correct.

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
the training result with 

-------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------
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
```> 
<br>
### --------------==================version 1 2021-08========------------------------------------ <br>
##### dataset used is voc2007, processed dataset can download from [processedvoc20072012](https://share.weiyun.com/NLjLT13V), unzip it to datas directory，[voc pretrained model](https://share.weiyun.com/7sTyVd7N)<br> 
```
#run
python models/train.py
#predict
python predict.py
```
<br>

## Reference
[https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)<br>
[https://github.com/Peterisfar/YOLOV3](https://github.com/Peterisfar/YOLOV3)<br>
[https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)<br>
[https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)<br>
[https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)<br>
[https://github.com/YunYang1994/tensorflow-yolov3](https://github.com/YunYang1994/tensorflow-yolov3)<br>
[https://github.com/mystic123/tensorflow-yolo-v3](https://github.com/mystic123/tensorflow-yolo-v3)<br>
[https://github.com/Ray-Luo/YOLOV3-PyTorch](https://github.com/Ray-Luo/YOLOV3-PyTorch)<br>
[https://github.com/DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)
