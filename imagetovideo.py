import os
import cv2
from tqdm import tqdm
 
def image2video():
    imagespath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\tracking\result\man'
    outputpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\tracking\result\man.avi'
    fps = 20
    lis = os.listdir(imagespath) #[:1000]
    img = cv2.imread(os.path.join(imagespath, lis[0]))
    h, w, c = img.shape
    sizes = (w, h) #w, h
    videowriter = cv2.VideoWriter(outputpath, cv2.VideoWriter_fourcc(*'XVID'), fps, sizes)
    lis = sorted(lis.__iter__(), key = lambda k:int(k.split(".")[0]))
    for i in tqdm(lis, mininterval=3):
        img = cv2.imread(os.path.join(imagespath, i))
        img = cv2.resize(img, sizes)
        videowriter.write(img)
image2video()