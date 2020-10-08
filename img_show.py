import numpy as np
import matplotlib.pyplot as plt
from os.path import getsize
def img_show(img_path):
    img = plt.imread(img_path)
    size = getsize(img_path)
    if size <1024:
        size = str(size)+' B'
    elif size>=1024 and size<1024**2:
        size = str(round(size/1024, 2))+' KB'
    elif size>=1048576:
        size = str(round(size/1024**2, 2))+' MB'
    print(r"This img's size is", img.shape[1], '(w)*', img.shape[0], '(h)', r", and the file size is ", size)
    plt.imshow(img)
    plt.show()
    return