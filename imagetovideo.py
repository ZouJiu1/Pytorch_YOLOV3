import os
import cv2
from tqdm import tqdm
 
imagespath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\imshow'
outputpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\kk.avi'
fps = 3
sizes = (32*10, 32*10) #w, h
videowriter = cv2.VideoWriter(outputpath, cv2.VideoWriter_fourcc(*'XVID'), fps, sizes)
lis = os.listdir(imagespath)
lis = sorted(lis.__iter__(), key = lambda k:int(k.split(".")[0]))
for i in tqdm(lis, mininterval=3):
    img = cv2.imread(os.path.join(imagespath, i))
    img = cv2.resize(img, sizes)
    videowriter.write(img)