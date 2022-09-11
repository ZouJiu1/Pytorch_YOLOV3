import xml.etree.ElementTree as ET
import pickle
import os
import shutil
from os import listdir, getcwd
from os.path import join
import numpy as np

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


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

def convert_annotation(in_file, out_file, transform):   #(year, image_id):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open(out_file, 'w')
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
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        if transform:
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        else:
            bb = b
            out_file.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')

def preprocess(inpath, outpath, names, transform):
    jpgpath = os.path.join(outpath, names, 'JPEGImages')
    labelpath = os.path.join(outpath, names, 'labels')

    if not os.path.exists(labelpath):
        os.makedirs(labelpath)
    if not os.path.exists(jpgpath):
        os.makedirs(jpgpath)

    for i in inpath:
        JPEGS = os.path.join(i, 'JPEGImages')
        xml   = os.path.join(i, 'Annotations')
        for j in os.listdir(xml):
            xmlpath = os.path.join(xml, j)
            JPEGimg = os.path.join(JPEGS, j.replace('xml', 'jpg'))

            out_file = os.path.join(labelpath, j.replace('xml', 'txt'))
            convert_annotation(xmlpath, out_file, transform)
            shutil.copyfile(JPEGimg, os.path.join(jpgpath, j.replace('xml', 'jpg')))
    
    lis = os.listdir(jpgpath)
    with open(os.path.join(outpath, names, '%s.txt'%names), 'w') as obj:
        for i in lis:
            obj.write(os.path.join(jpgpath, i)+'\n')

if __name__ == '__main__':
    basepath = r'/home/Pytorch_YOLOV3\\'
    outpath = basepath + r'datas'
    
    ##VOCtrainval_06-Nov-2007.tar,  VOCtrainval_11-May-2012.tar,  VOC2012test.tar
    trainpath = [basepath + r'datas\trainsets\VOCdevkit\VOC2007', 
             basepath + r'datas\trainsets\VOCdevkit\VOC2012']
    preprocess(trainpath, outpath, 'train', True)

    valpath = [basepath + r'datas\trainsets\VOCtest_06-Nov-2007\VOCdevkit\VOC2007'] #VOCtest_06-Nov-2007.tar
    preprocess(valpath, outpath, 'valid', False)