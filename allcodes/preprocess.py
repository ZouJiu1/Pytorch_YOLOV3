import xml.etree.ElementTree as ET
import pickle
import os
import shutil
from os import listdir, getcwd
from os.path import join
import numpy as np

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ['person', 'car'] #because of the absent of computer hardware

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(in_file, out_file, transform, truthpath):   #(year, image_id):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    col = []
    truth = []
    labels = []
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        dif = obj.find('difficult')
        difficult = 0
        if dif:
            difficult = dif.text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        labels.append(cls_id)
        dic[cls] += 1
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        # if transform:
        bb = convert((w,h), b)
        col.append(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        # else:
        bb = b
        truth.append(cls + " " + " ".join([str(a) for a in bb]) + '\n')
    
    if dic['person'] > 9000 and len(col) > 0 and 1 not in labels:
        return False

    if len(col) > 0:
        out_file = open(out_file, 'w')
        for j in col:
            out_file.write(j)
        out_file.close()
        truthpath = open(truthpath, 'w')
        for j in truth:
            truthpath.write(j)
        truthpath.close()
        return True
    return False


def preprocess(inpath, outpath, names, transform):
    jpgpath = os.path.join(outpath, names, 'JPEGImages')
    labelpath = os.path.join(outpath, names, 'labels')
    truth = os.path.join(outpath, names, 'truth')

    if not os.path.exists(labelpath):
        os.makedirs(labelpath)
    if not os.path.exists(jpgpath):
        os.makedirs(jpgpath)
    if not os.path.exists(truth):
        os.makedirs(truth)

    for i in inpath:
        JPEGS = os.path.join(i, 'JPEGImages')
        xml   = os.path.join(i, 'Annotations')
        for j in os.listdir(xml):
            if dic['person'] > 9000 and dic['car'] > 9000:
                break
            xmlpath = os.path.join(xml, j)
            JPEGimg = os.path.join(JPEGS, j.replace('xml', 'jpg'))

            out_file = os.path.join(labelpath, j.replace('xml', 'txt'))
            truthpath = os.path.join(truth, j.replace('xml', 'txt'))
            con = convert_annotation(xmlpath, out_file, transform, truthpath)
            if con==True:
                shutil.copyfile(JPEGimg, os.path.join(jpgpath, j.replace('xml', 'jpg')))
    
    lis = os.listdir(jpgpath)
    with open(os.path.join(outpath, names, '%s.txt'%names), 'w') as obj:
        for i in lis:
            obj.write(os.path.join(jpgpath, i)+'\n')
    
    di = {i:0 for i in classes}
    for i in os.listdir(labelpath):
        if 'cla' in i: continue
        with open(os.path.join(labelpath, i), 'r') as obj:
            for j in obj.readlines():
                if transform:
                    di[classes[int(j.split(' ')[0])]] += 1
                else:
                    di[j.split(' ')[0]] += 1
    print(di)

def splittrainval(basepath):
    jpgpth = os.path.join(basepath, 'train', 'JPEGImages')
    labelpth = os.path.join(basepath, 'train', "labels")
    truthpth = os.path.join(basepath, 'train', 'truth')

    valjpgpth = os.path.join(basepath, 'val', 'JPEGImages')
    vallabelpth = os.path.join(basepath, 'val', "labels")
    valtruthpth = os.path.join(basepath, 'val', 'truth')
    pth = [valjpgpth, vallabelpth, valtruthpth]
    for i in pth:
        if not os.path.exists(i):
            os.makedirs(i)
    kk = os.listdir(jpgpth)
    np.random.shuffle(kk)
    kk1 = kk[:200]
    kk2 = kk[200:]
    for i in kk1:
        txt = i.replace("jpg", 'txt')
        shutil.move(os.path.join(jpgpth, i), os.path.join(valjpgpth, i))
        shutil.move(os.path.join(labelpth, txt), os.path.join(vallabelpth, txt))
        shutil.move(os.path.join(truthpth, txt), os.path.join(valtruthpth, txt))
    with open(os.path.join(basepath, 'train.txt'), 'w') as obj:
        for i in kk2:
            obj.write(os.path.join(jpgpth, i)+'\n')
    with open(os.path.join(basepath, 'val.txt'), 'w') as obj:
        for i in kk1:
            obj.write(os.path.join(valjpgpth, i)+'\n')

if __name__ == '__main__':
    basepath = r'C:\Users\ZouJiu\Desktop\Pytorch_YOLOV3\\'
    outpath = basepath + r'datas'
    
    ##VOCtrainval_06-Nov-2007.tar,  VOCtrainval_11-May-2012.tar,  VOC2012test.tar
    # dic = {i:0 for i in classes}
    # trainpath = [basepath + r'datas\trainsets\VOCdevkit\VOC2007', 
    #          basepath + r'datas\trainsets\VOCdevkit\VOC2012',
    #          basepath + r'datas\trainsets\VOCtest_06-Nov-2007\VOCdevkit\VOC2007']
    # preprocess(trainpath, outpath, 'train', True)

    splittrainval(outpath)

    # dic = {i:0 for i in classes}
    # valpath = [basepath + r'datas\trainsets\VOCtest_06-Nov-2007\VOCdevkit\VOC2007'] #VOCtest_06-Nov-2007.tar
    # preprocess(valpath, outpath, 'valid', False)
