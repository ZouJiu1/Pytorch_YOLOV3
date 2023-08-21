import numpy as np
from multiprocessing import cpu_count

np.set_printoptions(formatter={'all':lambda x: str(x)}, suppress=False)

# def calsum():
#     sum = 1
#     array = [1]
#     print(1)
#     for i in range(98):
#         power_2 = 2**i
#         print("2^{}={}".format(i, power_2))
#         sum += power_2
#         array.append(power_2)
#     array.append(sum)
#     sum += sum
#     print("2^{}={}".format(98, 2**98))
#     print("array: ", array)
#     print("sum: ", sum) #2**99
#     for i in range(100):
#         mi = 100 - i
#         if i==0:
#             print("sum 整除", array[i], "=", sum==array[i] * (2**99))
#         else:
#             print("sum 整除", array[i], "=", sum==array[i] * (2**mi))

def calsum():
    sum = 1
    array = [1]
    print(1)
    for i in range(98):
        power_2 = 2*3**i
        print("2*3^{}={}".format(i, power_2))
        sum += power_2
        array.append(power_2)
    array.append(3**98)
    sum += 3**98
    print("3^98=", 3**98)
    print("array: ", array)
    print("sum: ", sum) #2**99
    for i in range(100):
        mi = 100 - i
        if i==0:
            print("sum 整除", array[i], "=", sum==array[i] * (2**99))
        else:
            print("sum 整除", array[i], "=", sum==array[i] * (2**mi))

def showcv():
    import cv2
    import json
    import os
    valpth = os.path.join(r'/root/project/Pytorch_YOLOV3/datas', 'instances_val2017.json')

    with open(valpth, 'r') as obj:
        jf = json.load(obj)

    for i in range(len(jf['images'])):
        nam = jf['images'][i]['file_name']
        if nam == '000000000139.jpg':
            width = jf['images'][i]['width']
            height = jf['images'][i]['height']
            break
        
    inpath = r'/root/autodl-tmp/val2017/000000000139.jpg'
    nam = '/root/project/Pytorch_YOLOV3/datas/cocoval/truth/000000000139.txt'
    img = cv2.imread(inpath)
    cvfont = cv2.FONT_HERSHEY_SIMPLEX
    kk = []
    with open(nam, 'r') as obj:
        for i in obj.readlines():
            i = i.strip().split(",")
            kk.append([i[0], float(i[1]), float(i[2]), float(i[2+1]), float(i[2*2])])
    for label, xmin, ymin, xmax, ymax in kk:
        xmin, ymin, xmax, ymax = int(xmin * (32*16) / width), int(ymin *(32*16)  /height ), int(xmax * (32*16) / width), int(ymax *(32*16) / height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [255, 0, 0], 2)
        cv2.putText(img, label, (xmin, ymin+13), cvfont, 0.5, [255, 0, 0], 1)
    cv2.imwrite(r'/root/project/Pytorch_YOLOV3/kk.jpg', img)

def filechange():
    import os
    pth = r'/root/project/yolov3tiny/2023-08-04yolov3tiny'
    for i in os.listdir(pth):
        os.remove(os.path.join(pth, i))
        os.system("touch %s"%os.path.join(pth, i))

def torchunique():
    # import torch
    # tensor = [0, 6, 6, 7, 2, 2, 1, 1, 2, 3, 6, 6, 9, 9, 1]
    # kk = torch.tensor(tensor)      # , range(len(tensor))
    # kkk = torch.unique(kk, sorted=False)
    # kkk = torch.unique_consecutive(kk)
    # r = 0
    import matplotlib.pyplot as plt
    import math
    x = np.arange(1, 101)
    finalr = 0.01
    baselr = 0.01
    
    y = [baselr*(1+(finalr - 1)*(i - 1)/(100 - 1)) for i in x]
    yk = [baselr*(((1 - math.cos(i * math.pi / 100)) / 2) * (finalr - 1) + 1) for i in x]
    lf = lambda x: (1 - x / 100) * (1.0 - 0.01) + 0.01
    decay = lambda x: 0.999 * (1 - math.exp(-x / 2000))
    x = np.arange(100)
    y = [lf(i) for i in x]
    plt.plot(x, y)
    plt.show()
    # (1+(0.01-1)*99/99) * 0.01

def autograd():
    import torch
    m = torch.arange(1, 16+1, dtype=torch.float).reshape((2*2, 2*2)).requires_grad_(True)
    y = m**2 + m * 2
    y.retain_grad()
    m.retain_grad()
    dy_m = torch.autograd.grad(y, m, grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)
    dy_m = dy_m[0]
    dy_m0 = dy_m[:, :2]
    k = 0

import os
# from multiprocessing import Pool
def removek(ik):
    try:
        os.remove(ik)    
    except:
        pass

def delelte():
    inpath = r'D:\backup\programming\dataset\train2017'
    # p = Pool(2*2+2)
    # for i in os.listdir(inpath):
    #     p.apply_async(removek, args=(os.path.join(inpath, i), ))
    # p.close()
    # p.join()
    import shutil
    shutil.rmtree(inpath)
    
    pth = r'D:\backup\programming\dataset\val2017'
    shutil.rmtree(pth)

import scipy
import lap

def repack():
    inputs = np.arange(9)
    np.random.shuffle(inputs)
    kk = np.reshape(inputs, (3, 3))
    cost_matrix = np.array([[5, 2, 0], [3, 7, 6], [1, 6, 9]])
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=100)
    x, y = scipy.optimize.linear_sum_assignment(cost_matrix)  # row x, col y
    return kk

if __name__=='__main__':
    # k = cpu_count()
    # kk = 0
    # calsum()
    # showcv()
    # filechange()
    # torchunique()
    # delelte()
    # autograd()
    repack()