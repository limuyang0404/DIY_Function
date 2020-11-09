# coding=UTF-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
import random
from os.path import isfile, join, getsize
'''this function is made to get a Two-dimensional dimensions with mirrored edges'''
# b = []
# # with open('color_code.txt', 'r') as f:
# #     line = f.readline()
# #     print(line)
# #     a.append(line)
# # print(a)
# file = open('color_code.txt')
# data = file.read().splitlines()
# for i in range(len(data)):
#     a = data[i].split()
#     a = list(map(int, a))
#     print(a)
#     b.append(a)
# print(b)
def color_code(filename):
    color_code = []
    data = open(filename).read().splitlines()
    for i in range(len(data)):
        line = data[i].split()
        line = list(map(int, line))
        color_code.append(line)
    return color_code
a = color_code('color_code.txt')
print(a)
# print(list(map(int, data)))

