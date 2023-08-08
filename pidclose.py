import os
abspath = os.path.abspath(__file__)
filename = os.sep.join(abspath.split(os.sep)[-2:])
abspath = abspath.replace(filename, "")
import sys
sys.path.append(abspath)

def closepid():
    pth = os.path.join(abspath, "pid.txt")
    os.system(r"ps -aux|grep train_yolov3 |tee %s" % pth)
    kk = []
    with open(pth, 'r') as obj:
        for i in obj.readlines():
            if 'grep' in i:
                continue
            i = i.strip()
            i = i.split(" ")
            while '' in i:
                i.remove('')
            kk.append(int(i[1]))
    for i in kk:
        try:
            os.system("kill -9 %d"%i)
        except:
            pass
    for i in kk:
        try:
            os.system("kill -9 %d"%i)
        except:
            pass

if __name__=="__main__":
    closepid()
            