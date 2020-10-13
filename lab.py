import numpy as np
import pandas as pd
import time
import decimal
import struct
import segyio
import os
from PIL import Image
from collections import Counter


def read_river(filename):
    '''filename should be the path of a binary file which have 5 columns representing Inline,Xline,X,Y,time'''
    datain = np.loadtxt(filename)  # read river file
    print('datain-info', type(datain), datain.shape)
    df = pd.DataFrame(data=datain, columns=['INLINE', 'XLINE', 'X', 'Y', 'TIME'])
    df['TIME'] = df['TIME'].apply(lambda x: round(x))  # make all time rounded
    df[['INLINE', 'XLINE', 'X', 'Y']] = df[['INLINE', 'XLINE', 'X', 'Y']].astype('int32')
    print('river df:\n', df)
    return df  # a dataframe of inline, xline, source coordination X and Y ,time


def river_position(river_dataframe):  # get the xyz interval of the river
    source_X_min = river_dataframe['X'].min()
    source_X_max = river_dataframe['X'].max()
    source_Y_min = river_dataframe['Y'].min()
    source_Y_max = river_dataframe['Y'].max()
    time_min = river_dataframe['TIME'].min()
    time_max = river_dataframe['TIME'].max()
    print('source_X_min, source_X_max, source_Y_min, source_Y_max, time_min, time_max=\n',
          source_X_min, source_X_max, source_Y_min, source_Y_max, time_min, time_max)
    return source_X_min, source_X_max, source_Y_min, source_Y_max, time_min, time_max


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
def segy_cut(filename, df):
    data = segyio.tools.cube(filename)
    label = segyio.tools.cube(filename)
    label[:, :, :] = 0
    print('***********************************')
    print('data:', data, "it's shape:", data.shape)
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    print('data:', label, "it's shape:", label.shape)
    # output_data = data[150:506, 3:465, 628:789]  # cut from segy file
    output_data = data[150:506, 3:465, 550:850]
    output_label = label[150:506, 3:465, 628:789]
    # dataframe = df.values
    dataframe = df.to_numpy()
    red = [237, 29, 37]
    blue = [0, 0, 147]
    # for i in range(38278):
    for i in range(dataframe.shape[0]):
        inline = dataframe[i][0]
        xline = dataframe[i][1]
        time = dataframe[i][4]    #the time have been rouned at read_river
        print('time is :\n', time, i)
        output_label[inline - 1350, xline - 1192, time - 1228] = 1
    for i in range(356):
        # slice = 255 * output_label[25 * i, :, :]
        slice = output_label[i, :, :]
        slice = np.moveaxis(slice, -1, 0)#time is the column, crossline is the row
        r = slice[:, :]*1
        g = slice[:, :]*1
        b = slice[:, :]*1
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        img = np.concatenate((r, g, b), axis=2)
        r1shape = img[:, :, 0].shape
        r1 = img[:, :, 0].flatten()
        for j in range(len(r1)):
            if r1[j] == 1:
                r1[j] = red[0]
            if r1[j] == 0:
                r1[j] = blue[0]
        img[:, :, 0] = r1.reshape(r1shape)
        g1shape = img[:, :, 1].shape
        g1 = img[:, :, 1].flatten()
        for j in range(len(g1)):
            if g1[j] == 1:
                g1[j] = red[1]
            if g1[j] == 0:
                g1[j] = blue[1]
        img[:, :, 1] = g1.reshape(g1shape)
        b1shape = img[:, :, 2].shape
        b1 = img[:, :, 2].flatten()
        for j in range(len(b1)):
            if b1[j] == 1:
                b1[j] = red[2]
            if b1[j] == 0:
                b1[j] = blue[2]
        img[:, :, 2] = b1.reshape(b1shape)
        print('img = \n', img.shape)
        img = np.array(img)
        img = np.asarray(img, np.uint8)
        im = Image.fromarray(img)
        im = img.convert('RGB')
        im.save('large/'+'inline_'+str(1350 + i) + '.png')
    for i in range(1, 101):
        # slice = 255 * output_label[25 * i, :, :]
        slice = output_label[:, 2*i, :]
        slice = np.moveaxis(slice, -1, 0)
        r = slice[:, :]*1
        g = slice[:, :]*1
        b = slice[:, :]*1
        r = np.expand_dims(r, axis=2)
        g = np.expand_dims(g, axis=2)
        b = np.expand_dims(b, axis=2)
        img = np.concatenate((r, g, b), axis=2)
        r1shape = img[:, :, 0].shape
        r1 = img[:, :, 0].flatten()
        for j in range(len(r1)):
            if r1[j] == 1:
                r1[j] = red[0]
            if r1[j] == 0:
                r1[j] = blue[0]
        img[:, :, 0] = r1.reshape(r1shape)
        g1shape = img[:, :, 1].shape
        g1 = img[:, :, 1].flatten()
        for j in range(len(g1)):
            if g1[j] == 1:
                g1[j] = red[1]
            if g1[j] == 0:
                g1[j] = blue[1]
        img[:, :, 1] = g1.reshape(g1shape)
        b1shape = img[:, :, 2].shape
        b1 = img[:, :, 2].flatten()
        for j in range(len(b1)):
            if b1[j] == 1:
                b1[j] = red[2]
            if b1[j] == 0:
                b1[j] = blue[2]
        img[:, :, 2] = b1.reshape(b1shape)
        img = np.array(img)
        img = np.asarray(img, np.uint8)
        im = Image.fromarray(img)
        # im = img.convert('RGB')
        im.save('large/'+'crossline_'+str(1220 + 2 * i) + '.png')
    for i in range(1, 11):
        slice = 255 * output_label[:, 30 * i, :]
        slice = np.moveaxis(slice, -1, 0)
        im = Image.fromarray(slice)
        im = im.convert('L')
        im.save('c' + str(1192 + 30 * i) + '.png')
    print('###########################################')
    print('output_data:', output_data, "it's shape:", output_data.shape)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    output_label = output_label.flatten()
    print(Counter(output_label))
    print('output_label:', output_label, "it's shape:", output_label.shape)
    output_data = output_data.flatten()
    print('###########################################')
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

df = read_river('CB27_river.dat')
# print('dataframe.type', type(df['TIME'][0]))#CB27_river.dat is the file of river
print('df:\n', df)
# source_X_min, source_X_max, source_Y_min, source_Y_max, time_min, time_max = river_position(df)
# segy_cut('cb27_600-2000ms_final.sgy', df)
# check_image('output.bin', datashape=(356, 462, 161))
# rawdata = np.fromfile('cut.bin', dtype='float32', count=-1, sep='').reshape(356,462,161)  #read saved cut segy file
#
# print(rawdata)
