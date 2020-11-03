import numpy as np
import pandas as pd
import time
import decimal
import struct
import segyio
import os
from PIL import Image
from collections import Counter
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
def label_location_read(filename):
    '''filename should be the path of a binary file which have 5 columns representing Inline,Xline,X,Y,time'''
    datain = np.loadtxt(filename)  # read river file
    label_location_df = pd.DataFrame(data=datain, columns=['INLINE', 'XLINE', 'X', 'Y', 'TIME'])
    label_location_df['TIME'] = label_location_df['TIME'].apply(lambda x: round(x))  # make all time rounded
    label_location_df[['INLINE', 'XLINE', 'X', 'Y']] = label_location_df[['INLINE', 'XLINE', 'X', 'Y']].astype('int32')
    return label_location_df  # a dataframe of inline, xline, source coordination X and Y ,time

def grayscale_into_rgbgrayscale(grayscale_img, channel):
    red = grayscale_img[:, :]*1
    green = grayscale_img[:, :]*1
    blue = grayscale_img[:, :]*1
    alpha = grayscale_img[:, :]*1
    red = np.expand_dims(red, axis=2)
    green = np.expand_dims(green, axis=2)
    blue = np.expand_dims(blue, axis=2)
    alpha = np.expand_dims(alpha, axis=2)
    if channel == 3:
        img = np.concatenate((red, green, blue), axis=2)
    print(img, type(img), img.shape)
    # img = np.moveaxis(img, -1, 0)
    # plt.imshow(img)
    # plt.show()
    return img


def river_position(river_dataframe):  # get the xyz interval of the river
    source_iline_min = river_dataframe['INLINE'].min()
    source_iline_max = river_dataframe['INLINE'].max()
    source_xline_min = river_dataframe['XLINE'].min()
    source_xline_max = river_dataframe['XLINE'].max()
    time_min = river_dataframe['TIME'].min()
    time_max = river_dataframe['TIME'].max()
    print('source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max=\n',
          source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max)
    return source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max


def segy_cutinfo(filename, Xmin, Xmax, Ymin, Ymax, timemin, timemax, inline_num, crossline_num):
    rawdata = open(filename, 'rb')
    segyheader = rawdata.read(3600)
    every_trace_simples = struct.unpack('>h', segyheader[3220:3222])[0]  # 每一道的采样点数
    sample_time = struct.unpack('>h', segyheader[3216:3218])[0]  # 采样时间（微秒）
    sample_time = sample_time / 1000
    xmax_line = 0
    xmin_line = 0
    ymax_line = 0
    ymin_line = 0
    for i in range(1, inline_num * crossline_num + 1):
        every_trace = rawdata.read(240)
        every_trace_x = struct.unpack('>l', every_trace[72:76])[0]  # 震源x坐标
        every_trace_y = struct.unpack('>l', every_trace[76:80])[0]  # 震源y坐标
        if (every_trace_x == Xmin and every_trace_y == Ymin):
            xmin_line = struct.unpack('>l', every_trace[8:12])[0]
            ymin_line = struct.unpack('>l', every_trace[20:24])[0]
        if (every_trace_x == Xmax and every_trace_y == Ymax):
            xmax_line = struct.unpack('>l', every_trace[8:12])[0]
            ymax_line = struct.unpack('>l', every_trace[20:24])[0]
        # if (every_trace_x == 658725 and every_trace_y == 4242650):
        #     x_line = struct.unpack('>l', every_trace[8:12])[0]
        #     y_line = struct.unpack('>l', every_trace[20:24])[0]
        trace_value = rawdata.read(every_trace_simples * 4)  # xyz坐标对应的具体数值
    # for j in range(1,every_trace_simples+1):
    #         every_trace_z = 1000+j*sample_time
    #         every_trace_value = struct.unpack('>I', trace_value[4*(j-1):4*j])[0]
    #         every_trace_value_true = ibm2ieee(every_trace_value)
    #         the_table[(i-1)*every_trace_simples+j][0] = every_trace_x
    #         the_table[(i - 1) * every_trace_simples + j][1] = every_trace_y
    #         the_table[(i - 1) * every_trace_simples + j][2] = every_trace_z
    #         the_table[(i - 1) * every_trace_simples + j][3] = every_trace_value_true
    #     printProgressBar(i, inline_num*crossline_num, prefix='process:', suffix='complete', length=100)
    # print('sample time:', sample_time)
    # print('every trace simples:', every_trace_simples)
    # np.savetxt('xyz_and_class1.csv', the_table, fmt='%f', delimiter=',')
    # print('segy file infomation stored in: xyz_and_class1.csv', )
    # print('xminline,yminline,xmaxline,ymaxline=', x_line, y_line)
    rawdata.close()
    return xmin_line, xmax_line, ymin_line, ymax_line


# river_dataframe = read_river('CB27_river.dat')
# Xmin, Xmax, Ymin, Ymax, timemin, timemax = river_position(river_dataframe)
# print(Xmin, Xmax, Ymin, Ymax, timemin, timemax)
# xmin_line, xmax_line, ymin_line, ymax_line = segy_cutinfo('cb27_600-2000ms_final.sgy', Xmin, Xmax, Ymin, Ymax, timemin, timemax, inline_num=601, crossline_num=664)
# print('xmin_line, xmax_line, ymin_line, ymax_line=', xmin_line, xmax_line, ymin_line, ymax_line)
# print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
def label_creat(label_class, label_start, label_end, label_cube, output_path):
    red = [237, 29, 37]
    blue = [0, 0, 147]
    for i in range(label_end - label_start + 1):
        slice = []
        if label_class == 'iline':
            slice = label_cube[i, :, :]
        elif label_class == 'xline':
            slice = label_cube[:, i, :]
        elif label_class == 'time':
            slice = label_cube[:, :, i]
        slice = np.moveaxis(slice, -1, 0)  # time is the column, crossline is the row
        img = grayscale_into_rgbgrayscale(slice, 3)
        red_shape = img[:, :, 0].shape
        red_channel = img[:, :, 0].flatten()
        for j in range(len(red_channel)):
            if red_channel[j] == 1:
                red_channel[j] = red[0]
            if red_channel[j] == 0:
                red_channel[j] = blue[0]
        img[:, :, 0] = red_channel.reshape(red_shape)
        green_shape = img[:, :, 1].shape
        green_channel = img[:, :, 1].flatten()
        for j in range(len(green_channel)):
            if green_channel[j] == 1:
                green_channel[j] = red[1]
            if green_channel[j] == 0:
                green_channel[j] = blue[1]
        img[:, :, 1] = green_channel.reshape(green_shape)
        blue_shape = img[:, :, 2].shape
        blue_channel = img[:, :, 2].flatten()
        for j in range(len(blue_channel)):
            if blue_channel[j] == 1:
                blue_channel[j] = red[2]
            if blue_channel[j] == 0:
                blue_channel[j] = blue[2]
        img[:, :, 2] = blue_channel.reshape(blue_shape)
        # print('img = \n', img.shape)
        img = np.array(img)
        img = np.asarray(img, np.uint8)
        im = Image.fromarray(img)
        im = im.convert('RGB')
        print('%^%^%^%^%^%^%^%^%^%^%^%^%^')
        im.save(output_path + '\\' + label_class + '_' + str(label_start + i) + '.png')
    return

def segy_cut(filename, df, slice_class='iline'):
    label_iline_start, label_iline_end, label_xline_start, label_xline_end, label_time_start, label_time_end = \
        river_position(df)
    print(label_iline_start, label_iline_end, label_xline_start, label_xline_end, label_time_start, label_time_end)
    output_data = segyio.tools.cube(filename)    #origion seismic data
    output_label = segyio.tools.cube(filename)   #label data cube
    output_label[:, :, :] = 0
    data_iline_start, data_xline_start, data_time_start = get_location_start(filename)
    print('***********************************')
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    # output_label = label[:iline_end, xline_start - 1:xline_end, time_start-1:time_end]
    output_data = output_data[label_iline_start-data_iline_start:label_iline_end-data_iline_start+1, \
                  label_xline_start-data_xline_start:label_xline_end-data_xline_start+1, label_time_start-data_time_start:label_time_end-data_time_start+1]  # cut from segy file
    # output_data = data[150:506, 3:465, 550:850]
    print('point info:', label_iline_start-data_iline_start, label_iline_end-data_iline_start+1, label_xline_start-data_xline_start, \
          label_xline_end - data_xline_start + 1, label_time_start-data_time_start, label_time_end-data_time_start+1)
    output_label = output_label[label_iline_start-data_iline_start:label_iline_end-data_iline_start+1, \
                  label_xline_start-data_xline_start:label_xline_end-data_xline_start+1, label_time_start-data_time_start:label_time_end-data_time_start+1] #slice from source data cube
    # dataframe = df.values
    array_from_dataframe = df.to_numpy()
    # colour_list = [[237,29,37], [0,0,147]]
    red = [237, 29, 37]
    blue = [0, 0, 147]
    for i in range(array_from_dataframe.shape[0]):
        inline = array_from_dataframe[i][0]
        xline = array_from_dataframe[i][1]
        time = array_from_dataframe[i][4]    #the time have been rouned at read_river
        print('time is :\n', time, i)
        output_label[inline - label_iline_start, xline - label_xline_start, time - label_time_start] = 1
    if slice_class == 'iline' :
        label_creat('iline', label_iline_start, label_iline_end, output_label, 'test')
        for i in range(label_iline_end-label_iline_start+1):
            # slice = 255 * output_label[25 * i, :, :]
            slice_iline = output_label[i, :, :]
            slice_iline = np.moveaxis(slice_iline, -1, 0)#time is the column, crossline is the row
            img_iline = grayscale_into_rgbgrayscale(slice_iline, 3)
            red_shape = img_iline[:, :, 0].shape
            red_channel = img_iline[:, :, 0].flatten()
            for j in range(len(red_channel)):
                if red_channel[j] == 1:
                    red_channel[j] = red[0]
                if red_channel[j] == 0:
                    red_channel[j] = blue[0]
            img_iline[:, :, 0] = red_channel.reshape(red_shape)
            green_shape = img_iline[:, :, 1].shape
            green_channel = img_iline[:, :, 1].flatten()
            for j in range(len(green_channel)):
                if green_channel[j] == 1:
                    green_channel[j] = red[1]
                if green_channel[j] == 0:
                    green_channel[j] = blue[1]
            img_iline[:, :, 1] = green_channel.reshape(green_shape)
            blue_shape = img_iline[:, :, 2].shape
            blue_channel = img_iline[:, :, 2].flatten()
            for j in range(len(blue_channel)):
                if blue_channel[j] == 1:
                    blue_channel[j] = red[2]
                if blue_channel[j] == 0:
                    blue_channel[j] = blue[2]
            img_iline[:, :, 2] = blue_channel.reshape(blue_shape)
            # print('img = \n', img.shape)
            img_iline = np.array(img_iline)
            img_iline = np.asarray(img_iline, np.uint8)
            im_iline = Image.fromarray(img_iline)
            im_iline = im_iline.convert('RGB')
            print('%^%^%^%^%^%^%^%^%^%^%^%^%^')
            im_iline.save('test\inline_'+str(label_iline_start + i) + '.png')
    if slice_class == 'xline':
        for i in range(label_xline_end-label_xline_start+1):
            # slice = 255 * output_label[25 * i, :, :]
            slice_xline = output_label[:, i, :]
            slice_xline = np.moveaxis(slice_xline, -1, 0)  # slice(time,inline, cross_line)
            img_xline = grayscale_into_rgbgrayscale(slice_xline, 3)
            red_shape = img_xline[:, :, 0].shape
            red_channel = img_xline[:, :, 0].flatten()
            for j in range(len(red_channel)):
                if red_channel[j] == 1:
                    red_channel[j] = red[0]
                if red_channel[j] == 0:
                    red_channel[j] = blue[0]
            img_xline[:, :, 0] = red_channel.reshape(red_shape)
            green_shape = img_xline[:, :, 1].shape
            green_channel = img_xline[:, :, 1].flatten()
            for j in range(len(green_channel)):
                if green_channel[j] == 1:
                    green_channel[j] = red[1]
                if green_channel[j] == 0:
                    green_channel[j] = blue[1]
            img_xline[:, :, 1] = green_channel.reshape(green_shape)
            blue_shape = img_xline[:, :, 2].shape
            blue_channel = img_xline[:, :, 2].flatten()
            for j in range(len(blue_channel)):
                if blue_channel[j] == 1:
                    blue_channel[j] = red[2]
                if blue_channel[j] == 0:
                    blue_channel[j] = blue[2]
            img_xline[:, :, 2] = blue_channel.reshape(blue_shape)
            img_xline = np.array(img_xline)
            img_xline = np.asarray(img_xline, np.uint8)
            im_xline = Image.fromarray(img_xline)
            im_xline = im_xline.convert('RGB')
            im_xline.save('test\crossline_'+str(label_xline_start + i) + '.png')
    if slice_class == 'time':
        for i in range(label_time_end-label_time_start+1):
            # slice = 255 * output_label[25 * i, :, :]
            slice_time = output_label[:, :, i]
            slice_time = np.moveaxis(slice_time, -1, 0)#time is the column, crossline is the row
            img_time = grayscale_into_rgbgrayscale(slice_time, 3)
            red_shape = img_time[:, :, 0].shape
            red_channel = img_time[:, :, 0].flatten()
            for j in range(len(red_channel)):
                if red_channel[j] == 1:
                    red_channel[j] = red[0]
                if red_channel[j] == 0:
                    red_channel[j] = blue[0]
            img_time[:, :, 0] = red_channel.reshape(red_shape)
            green_shape = img_time[:, :, 1].shape
            green_channel = img_time[:, :, 1].flatten()
            for j in range(len(green_channel)):
                if green_channel[j] == 1:
                    green_channel[j] = red[1]
                if green_channel[j] == 0:
                    green_channel[j] = blue[1]
            img_time[:, :, 1] = green_channel.reshape(green_shape)
            blue_shape = img_time[:, :, 2].shape
            blue_channel = img_time[:, :, 2].flatten()
            for j in range(len(blue_channel)):
                if blue_channel[j] == 1:
                    blue_channel[j] = red[2]
                if blue_channel[j] == 0:
                    blue_channel[j] = blue[2]
            img_time[:, :, 2] = blue_channel.reshape(blue_shape)
            # print('img = \n', img.shape)
            img_time = np.array(img_time)
            img_time = np.asarray(img_time, np.uint8)
            im_time = Image.fromarray(img_time)
            im_time = im_time.convert('RGB')
            im_time.save(r'test\time_'+str(label_time_start + i) + '.png')
    print('output_data:', output_data, "it's shape:", output_data.shape)
    output_label = output_label.flatten()
    print(Counter(output_label))
    print('output_label:', output_label, "it's shape:", output_label.shape)
    output_data = output_data.flatten()
    print('output_data:', output_data, "it's shape:", output_data.shape)
    output_data.tofile("cut.bin")
    output_label.tofile('label.bin')
    return


# segy_cut('cb27_600-2000ms_final.sgy', river_dataframe)
def check_image(filename, datashape=(1, 2, 3)):
    data = np.memmap(filename, dtype='float32', mode='r')
    print('the data is:', data)
    print(Counter(data))
    data = data.reshape(datashape)
    data = np.array(data)
    print('!@#$%!@#$%!@#$%!@#$%')
    print('the data is:', data)
    print(type(data), data.shape)
    for i in range(1, 11):
        slice = 255 * data[25 * i, :, :]
        slice = np.moveaxis(slice, -1, 0)
        im = Image.fromarray(slice)
        im = im.convert('L')
        im.save(str(1350 + 25 * i) + '.png')

def get_location_start(file):
    with segyio.open(file) as f:
        line_start = f.ilines[0]
        xline_start = f.xlines[0]
        time_start = int(f.samples[0])
    print(line_start, xline_start, time_start)
    return line_start, xline_start, time_start
df = label_location_read('CB27_river.dat')
# print('dataframe.type', type(df['TIME'][0]))#CB27_river.dat is the file of river
img1 = plt.imread(r'C:\Users\Administrator\Desktop\PStest\908196604566533.jpg')
river_position(df)
print('df:\n', df)
print(img1.shape, img1[:, :, 0])
k_on = grayscale_into_rgbgrayscale(img1[:, :, 0], 3)
plt.imshow(k_on)
plt.show()
get_location_start(r"C:\Users\Administrator\Desktop\cb27_600-2000ms_final.sgy")
# source_X_min, source_X_max, source_Y_min, source_Y_max, time_min, time_max = river_position(df)
segy_cut(r'C:\Users\Administrator\Desktop\cb27_600-2000ms_final.sgy', df, slice_class='iline')
# check_image('output.bin', datashape=(356, 462, 161))
# rawdata = np.fromfile('cut.bin', dtype='float32', count=-1, sep='').reshape(356,462,161)  #read saved cut segy file
#
# print(rawdata)