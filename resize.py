import os
import cv2
from multiprocessing import cpu_count, Pool

def fun(i):
    img = cv2.imread(i)
    h, w, c = img.shape
    k = 32 * 16
    if h==k and w==k:
        return
    img = cv2.resize(img, (k, k))
    cv2.imwrite(i, img)

def process():
    inpath = r'/root/autodl-tmp'
    ww = []
    for r, d, f in os.walk(inpath):
        for i in f:
            if '.jpg' not in i:
                continue
            ww.append(os.path.join(r, i))
    p = Pool(cpu_count())
    for i in ww:
        # fun(i)
        p.apply_async(fun, args=(i,))
    p.close()
    p.join()

if __name__=="__main__":
    process()