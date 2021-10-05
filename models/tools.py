import os
import time

def log():
    path =r'C:\Users\10696\Desktop\yolov3\log'

    while True:
        lis = os.listdir(path)
        lis.sort()
        lis.remove('log.txt')
        print('lis[-1]: ', lis[-1])
        for i in lis[:-1]:
            if 'model' in i:
                print(i)
                os.remove(os.path.join(path, i))
        print('waiting......')
        time.sleep(600)

def filelist():
    path =r'C:\Users\10696\Desktop\Pytorch_YOLOV3myself\val2017'
    with open(r'C:\Users\10696\Desktop\Pytorch_YOLOV3myself\cocoval2017.txt', 'w') as f:
        for i in os.listdir(path):
            f.write(os.path.join(path,i)+'\n')

if __name__ == '__main__':
    # log()
    filelist()
    