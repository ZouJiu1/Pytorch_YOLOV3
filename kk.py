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

if __name__=='__main__':
    # k = cpu_count()
    # kk = 0
    # calsum()
    # showcv()
    filechange()