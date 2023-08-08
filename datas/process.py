import os
import json
import shutil
import numpy as np
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def prepro():
    inpath = r'C//Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/datas/VOCdevkit\labels'
    img = r'//Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/datas/VOCdevkit\JPEGImages'
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
    inpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\VOC2007copyss'
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
    traintxt= r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\VOC2007copyss\train.txt'
    lis = []
    with open(traintxt, 'r') as f:
        for i in f.readlines():
            i = i.strip()
            lis.append(i)
    np.random.shuffle(lis)
    np.random.shuffle(lis)
    lis = lis[:600]
    f = open(r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\train.txt', 'w')
    for i in lis:
        print(i)
        nam = i.split(os.sep)[-1]
        txt = i.replace("JPEGImages", 'labels').replace("jpg", "txt")
        shutil.copyfile(i, r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\JPEGImages\\'+nam)
        shutil.copyfile(txt, r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\labels\\'+nam.replace("jpg", "txt"))
        f.write(r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\JPEGImages\\'+nam+'\n')
    f.close()

def static():
    inpath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\datas\train\labels'
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

def parseprototxt():
    inpath = r'//Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/datas/220831prune37.prototxt'
    inpath = r'//Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/datas/yolofastest.prototxt'
    f = open(inpath, 'r')
    dic = {"group":1, 'pad':0}
    start = -9
    fil = open("//Pytorch_YOLOV3/Pytorch_YOLOV3_Jiu/datas/openfile_yolofastest.txt", 'w')
    precha = 3
    count = 1
    for i in f.readlines():
        i = i.strip().replace(" ", '')
        if 'name' in i:
            dic["name"] = str(i.split(":")[1].encode().decode())
        if "type" in i:
            dic["type"] = str(i.split(":")[1].encode().decode())
        if 'convolution_param' in i:
            start = 9
            continue
        if start > 0 and '}' in i:
            start = -9
            line = "self.block{} = ConvBlock_LN({}, {}, kernel_size = {}, stride = {}, padding = {}, groups = {}, bias = False)\n".format(count, precha, dic['num_output'], \
                    dic['kernel_size'], dic['stride'], dic['pad'], dic['group'])
            precha = dic["num_output"]
            fil.write(line)
            count += 1
            fil.flush()
            dic = {"group":1, 'pad':0}
            rel = -9
        if start > 0:
            if 'bias_term' in i:
                continue
            dic[i.split(":")[0]] = int(i.split(":")[-1])
    f.close()
    fil.close()

def getlabel():
    import cv2
    inpath = r'//yolov5-6.1/data/20221112_yolofastest/val.txt'
    outpath = r'//yolov5-6.1/data/20221112_yolofastest/val'
    classes = ['person', 'vehicle']
    with open(inpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            txt = i.replace("JPEGImages", 'labels').replace('jpg', 'txt')
            nam = txt.split(os.sep)[-1]
            ff = open(os.path.join(outpath, nam), 'w')
            print(i)
            img = cv2.imread(i)
            h, w, _ = img.shape
            with open(txt, 'r') as oj:
                for j in oj.readlines():
                    j = j.strip()
                    label, ncx, ncy, nw, nh = j.split(' ')
                    label, ncx, ncy, nw, nh = int(label), float(ncx), float(ncy), float(nw), float(nh)
                    xmin = (ncx - nw/2)*w
                    xmax = (ncx + nw/2)*w
                    ymin = (ncy - nh/2)*h
                    ymax = (ncy + nh/2)*h
                    kl = [classes[int(label)], str(xmin), str(ymin), str(xmax), str(ymax)]
                    ff.write(" ".join(kl)+'\n')
            ff.close()

def check():
    inpath = r'//yolov5-6.1/data/20220826_night/20220826train.txt'
    kee = []
    with open(inpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            if os.path.exists(i):
                kee.append(i)
    with open(inpath, 'w') as obj:
        for i in kee:
            obj.write(i+'\n')

def getmax():
    inpath = r'//yolov5-6.1/data/20221112_yolofastest/train.txt'
    kl = []
    with open(inpath, 'r') as obj:
        for i in obj.readlines():
            i = i.strip()
            cnt = 0
            with open(i.replace("JPEGImages",'labels').replace("jpg", 'txt'), 'r') as oj:
                for j in oj.readlines():
                    cnt+=1
            kl.append(cnt)
    print(np.max(kl))
            
if __name__=="__main__":
    # prepro()
    # sample()
    # static()
    # parseprototxt()
    # getlabel()
    # check()
    getmax()
