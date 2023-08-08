import os
import time
import numpy as np

def rmmodel():
    inpath = r'/root/project/yolov3tiny/2023-08-03yolov3tiny'
    lis = os.listdir(inpath)
    dic = {}
    for i in lis:
        num = i.split('_')[1]
        num = int(num[1:])
        dic[num] = os.path.join(inpath, i)
        if 'map' not in i:
            os.remove(os.path.join(inpath, i))
    dic = sorted(dic.items(), key = lambda x: x[0])
    if(len(dic) < 6):
        return None
    dic = dic[:-3]
    for i in dic:
        os.remove(i[1])
        os.system("touch %s " % (i[1]))

if __name__=="__main__":
    while True:
        rmmodel()
        time.sleep(200)