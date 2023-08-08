import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from multiprocessing import Array

ik = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\datas\1000000000000.jpg'
pth = r'F:\20230416\coco\images\train2014'
trainpath = r'C:\Users\10696\Desktop\Pytorch_YOLOV3\2023\PyTorch-YOLOv3-master\data\person\personcartrain.txt'

def get(img):
    img = img.flatten()
    img = (img - np.mean(img))/ np.std(img)
    return img

zih = (200, 200)
k0 = get(cv2.resize(cv2.imread(ik, 0), zih))
dic = {}

def getfil(imgpth, arr, num):
    img = cv2.imread(imgpth, 0)
    img = cv2.resize(img, zih)
    kl = get(img)
    cok = np.sum(kl * k0) / (np.sqrt(np.sum(k0 * k0)) * np.sqrt(np.sum(kl * kl)))
    arr[num] = cok

# for i in pth:
#     img = cv2.imread(os.path.join(pth, i), 0)
#     img = cv2.resize(img, zih)
#     kl = get(img)
#     cok = np.sum(kl * k0) / (np.sqrt(k0 * k0) * np.sqrt(kl * kl))
#     dic[i] = cok
if __name__=='__main__':
    p = Pool(cpu_count()//2)
    kee = []
    with open(trainpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            kee.append(i)
    num = 0
    arr = Array('f', len(kee))
    for i in kee:
        getfil(i, arr, num,)
        # p.apply_async(getfil, args=(i, arr, num, ))
        num += 1
    p.close()
    p.join()
    p.terminate()

    num = 0
    for i in kee:
        dic[i] = arr[num]
        num += 1
    nh = sorted(dic.items(), key = lambda x : x[1], reverse = True)
    print(nh[:10])