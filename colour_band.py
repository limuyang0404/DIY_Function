# coding=UTF-8
import numpy as np
from PIL import Image
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''
colour_class = [[0, 191, 255], [255, 20, 147], [255, 127, 36], [238, 130, 238], [0, 255, 255], [192, 255, 62], [255, 14, 246], [255, 255, 0], [255, 48, 48]]
img = np.zeros(shape=(500, 900, 3))
for i in range(9):
    img[:, i * 100:(i + 1) * 100, 0] = colour_class[i][0]
    img[:, i * 100:(i + 1) * 100, 1] = colour_class[i][1]
    img[:, i * 100:(i + 1) * 100, 2] = colour_class[i][2]
img = np.array(img)
img = np.asarray(img, np.uint8)
im = Image.fromarray(img)
im = im.convert('RGB')
im.save('test\colour_list.png')
