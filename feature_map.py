# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os

def feature_img(input_path, width_number, height_number, out_path, interval=10):
    '''
    A function designed to save same size picture in one picture.
    input_path:the input file's absolute path,
    width_number:the number of small pictures in the horizontal direction in the output.
    height_number:the number of small pictures in the vertically direction in the output.
    output_path:the output file'sabsolute path.
    interval:the interval between every small img.
    '''
    files = [f for f in listdir(input_path) if
             isfile(join(input_path, f))]
    img_height = (plt.imread(join(input_path, files[0]))).shape[0]
    img_width = (plt.imread(join(input_path, files[0]))).shape[1]
    img_channel = 3
    print(img_channel)
    # img_channel = (plt.imread(join(input_path, files[0]))).shape[2]
    output_img_height = height_number * img_height + (height_number+1)*interval
    output_img_width = width_number * img_width + (width_number+1)*interval
    if (plt.imread(join(input_path, files[0]))).ndim == 2:
        output_img = np.ones((output_img_height, output_img_width))*255
        img_counter = 0
        for i in range(height_number):
            for j in range(width_number):
                img = plt.imread(join(input_path, files[img_counter]))
                if files[img_counter][-3:] == 'png':
                    output_img[(i + 1) * interval+ i * img_height:(i + 1) * interval + (i+1) * img_height,
                    (j + 1) * interval + j * img_width:(j + 1) * interval + (j+1) * img_width] = img[:, :]*255
                elif files[img_counter][-3:] == 'jpg':
                    output_img[(i + 1) * interval + i * img_height:(i + 1) * interval + (i + 1) * img_height,
                    (j + 1) * interval + j * img_width:(j + 1) * interval + (j + 1) * img_width] = img[:, :]
                img_counter += 1
        im = Image.fromarray(np.uint16(output_img))
        im = im.convert('L')
        im.save(join(out_path, str(os.path.basename(os.path.normpath(input_path))) + '.png'))#the file name is the last file folder name of input file

    elif (plt.imread(join(input_path, files[0]))).ndim > 2:
        img_channel = (plt.imread(join(input_path, files[0]))).shape[2]
        output_img = np.ones((output_img_height, output_img_width, img_channel))*255
        img_counter = 0
        for i in range(height_number):
            for j in range(width_number):
                img = plt.imread(join(input_path, files[img_counter]))
                if files[img_counter][-3:] == 'png':
                    output_img[(i + 1) * interval+ i * img_height:(i + 1) * interval + (i+1) * img_height,
                    (j + 1) * interval + j * img_width:(j + 1) * interval + (j+1) * img_width, :] = img[:, :, :]*255
                elif files[img_counter][-3:] == 'jpg':
                    output_img[(i + 1) * interval + i * img_height:(i + 1) * interval + (i + 1) * img_height,
                    (j + 1) * interval + j * img_width:(j + 1) * interval + (j + 1) * img_width, :] = img[:, :, :]
                img_counter += 1
        im = Image.fromarray(np.uint8(output_img))
        im = im.convert('RGB')
        im.save(join(out_path, str(os.path.basename(os.path.normpath(input_path))) + '.png'))

    # output_img = np.zeros((output_img_height, output_img_width, img_channel))
    # img_counter = 0
    # for i in range(height_number):
    #     for j in range(width_number):
    #         img = plt.imread(join(input_path(input_path, files[img_counter])))
    #         output_img[(i+1)*interval:(i+1)*interval+i*img_height, (j+1)*interval:(j+1)*interval+j*img_width, :] = img[:, :, :]


    return



if __name__ == '__main__':
    # feature_img(r'C:\Users\Administrator\Desktop\DIY-function\3_94_890_feature_map\layer1', 8, 4, r'C:\Users\Administrator\Desktop\DIY-function')
    # feature_img(r'C:\Users\Administrator\Desktop\DIY-function\3_94_890_feature_map\layer2', 8, 8, r'C:\Users\Administrator\Desktop\DIY-function')
    # feature_img(r'C:\Users\Administrator\Desktop\DIY-function\3_94_890_feature_map\layer3', 16, 8, r'C:\Users\Administrator\Desktop\DIY-function')
    feature_img(r'C:\Users\Administrator\Desktop\DIY-function\11', 2, 1, r'C:\Users\Administrator\Desktop\DIY-function')
