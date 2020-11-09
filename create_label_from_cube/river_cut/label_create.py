# coding=UTF-8
import numpy as np
from PIL import Image
def color_code(filename):
    color_code = []
    data = open(filename).read().splitlines()
    for i in range(len(data)):
        line = data[i].split()
        line = list(map(int, line))
        color_code.append(line)
    return color_code

def grayscale_into_rgbgrayscale(grayscale_img, channel):
    red = grayscale_img[:, :]*1
    green = grayscale_img[:, :]*1
    blue = grayscale_img[:, :]*1
    alpha = grayscale_img[:, :]*1
    red = np.expand_dims(red, axis=2)
    green = np.expand_dims(green, axis=2)
    blue = np.expand_dims(blue, axis=2)
    alpha = np.expand_dims(alpha, axis=2)
    img = np.zeros(shape=(3, 4, 5))
    if channel == 3:
        img = np.concatenate((red, green, blue), axis=2)
    return img

def layer_replace(layer_class, label_origin, coloru_list):
    layer_class = ['red', 'green', 'blue'].index(layer_class)
    shape = label_origin.shape
    replaced_layer = label_origin.flatten()
    for i in range(len(replaced_layer)):
        for j in range(len(coloru_list)):
            if replaced_layer[i] == j:
                replaced_layer[i] = coloru_list[j][layer_class]
    replaced_layer = replaced_layer.reshape(shape)
    return replaced_layer

def label_create(label_class, label_start, label_end, label_cube, output_path):
    colour_class = color_code('color_code.txt')
    for i in range(label_end - label_start + 1):
        slice = np.zeros(shape=(3, 4, 5))
        if label_class == 'iline':
            slice = label_cube[i, :, :]
        elif label_class == 'xline':
            slice = label_cube[:, i, :]
        elif label_class == 'time':
            slice = label_cube[:, :, i]
        slice = np.moveaxis(slice, -1, 0)  # time is the column, crossline is the row
        img = grayscale_into_rgbgrayscale(slice, 3)
        img[:, :, 0] = layer_replace('red', img[:, :, 0], colour_class)
        img[:, :, 1] = layer_replace('green', img[:, :, 1], colour_class)
        img[:, :, 2] = layer_replace('blue', img[:, :, 2], colour_class)
        img = np.asarray(img, dtype=np.uint8)
        im = Image.fromarray(img)
        im = im.convert('RGB')
        im.save(output_path + '\\' + label_class + '_' + str(label_start + i) + '.png')
        print('\rThe ' + str(i+1)+'th ' + label_class + ' slice have been created!', end="")
    return