import os
import shutil 
import numpy as np
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def prepro():
    inpath = r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\VOC2007copyss\labels'
    img = r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\VOC2007copyss\JPEGImages'
    for i in os.listdir(img):
        if 'txt' in i:
            os.remove(os.path.join(img, i))
    se = set()
    for i in os.listdir(inpath):
        cnt = 0
        if 'x.txt' in i:
            continue
        f = open(os.path.join(inpath, 'x.txt'), 'w')
        with open(os.path.join(inpath, i), 'r') as obj:
            for j in obj.readlines():
                ind = int(j.split(" ")[0])
                if(classes[ind]=="person" or classes[ind]=="car"):
                    if classes[ind]=="person":
                        j = j[:3].replace(str(ind), '0') + j[3:]
                    else:
                        j = j[:3].replace(str(ind), '1') + j[3:]
                    f.write(j)
                    cnt += 1
        f.close()
        if cnt==0:
            os.remove(os.path.join(img, i.replace('txt', 'jpg')))
        os.remove(os.path.join(inpath, i))
        if cnt!=0:
            os.rename(os.path.join(inpath, 'x.txt'), os.path.join(inpath, i))
    x = ['test.txt', 'train.txt', 'val.txt']
    inpath = r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\VOC2007copyss'
    for i in x:
        if 'x.txt' in i:
            continue
        f = open(os.path.join(inpath, 'x.txt'), 'w')
        with open(os.path.join(inpath, i), 'r') as obj:
            for j in obj.readlines():
                j = j.strip()
                j = j.replace("VOC2007", "VOC2007copyss")
                if os.path.exists(j):
                    f.write(j+'\n')
        f.close()
        os.remove(os.path.join(inpath, i))
        os.rename(os.path.join(inpath, 'x.txt'), os.path.join(inpath, i)) 
    os.remove(os.path.join(inpath, 'x.txt'))

def sample():
    traintxt= r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\VOC2007copyss\train.txt'
    lis = []
    with open(traintxt, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            lis.append(i)
    np.random.shuffle(lis)
    np.random.shuffle(lis)
    lis = lis[:600]
    f = open(r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\train\train.txt', 'w')
    for i in lis:
        print(i)
        nam = i.split(os.sep)[-1]
        txt = i.replace("JPEGImages", 'labels').replace("jpg", "txt")
        shutil.copyfile(i, r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\train\JPEGImages\\'+nam)
        shutil.copyfile(txt, r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\train\labels\\'+nam.replace("jpg", "txt"))
        f.write(r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\train\JPEGImages\\'+nam+'\n')
    f.close()

def static():
    inpath = r'C:\Users\ZouJiu\Desktop\projects\Pytorch_YOLOV3\datas\train\labels'
    dic = {}
    for i in os.listdir(inpath):
        with open(os.path.join(inpath, i), 'r') as f:
            for j in f.readlines():
                j = j.strip().split()
                if j[0] not in dic.keys():
                    dic[j[0]] = 1
                else:
                    dic[j[0]] += 1
    print(dic)


if __name__=="__main__":
    # prepro()
    # sample()
    static()