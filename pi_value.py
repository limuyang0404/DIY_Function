# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
from collections import Counter
'''this function is made to predict pi'''
# img = plt.imread(r'C:\Users\Administrator\Desktop\DIY-function\circle.png')
# area = img[:, :, 2]
def calculation_pi(iterations_times):
    black = 0
    for i in range(1,iterations_times+1):
        x = random.random()
        y = random.random()
        if x**2+y**2<=1:
            black += 1
    pi = black/iterations_times*4
    print('The pi value is :', pi)
    return pi
# print(area, area.shape)
# plt.imshow(img)
# plt.show()

