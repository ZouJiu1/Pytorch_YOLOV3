import os
import time
import numpy as np

def rmmodel_717():
    inpath = r'/mnt/717'
    lis = os.listdir(inpath)
    dic = {}
    for i in lis:
        num = i.split('_')[1]
        num = int(num)
        dic[num] = os.path.join(inpath, i)
    dic = sorted(dic.items(), key = lambda x: x[0])
    if(len(dic) <= 2):
        return None
    dic = dic[:-2]
    for i in dic:
        os.remove(i[1])
        print(i[1])

def rmmodel_730():
    inpath = r'/mnt/730'
    lis = os.listdir(inpath)
    dic = {}
    for i in lis:
        num = i.split('_')[1]
        num = int(num)
        dic[num] = os.path.join(inpath, i)
    dic = sorted(dic.items(), key = lambda x: x[0])
    if(len(dic) <= 2):
        return None
    dic = dic[:-2]
    for i in dic:
        os.remove(i[1])
        print(i[1])

if __name__=="__main__":
    while True:
        rmmodel_717()
        rmmodel_730()
        time.sleep(60)