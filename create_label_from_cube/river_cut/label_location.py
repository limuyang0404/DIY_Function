# coding=UTF-8
import numpy as np
import pandas as pd


def label_location_read(filename):
    '''filename should be the path of a binary file which have 5 columns representing Inline,Xline,X,Y,time'''
    datain = np.loadtxt(filename)  # read river file
    label_location_df = pd.DataFrame(data=datain, columns=['INLINE', 'XLINE', 'X', 'Y', 'TIME'])
    label_location_df['TIME'] = label_location_df['TIME'].apply(lambda x: round(x))  # make all time rounded
    label_location_df[['INLINE', 'XLINE', 'X', 'Y']] = label_location_df[['INLINE', 'XLINE', 'X', 'Y']].astype('int32')
    return label_location_df  # a dataframe of inline, xline, source coordination X and Y ,time


def label_location(river_dataframe):  # get the xyz interval of the river
    source_iline_min = river_dataframe['INLINE'].min()
    source_iline_max = river_dataframe['INLINE'].max()
    source_xline_min = river_dataframe['XLINE'].min()
    source_xline_max = river_dataframe['XLINE'].max()
    time_min = river_dataframe['TIME'].min()
    time_max = river_dataframe['TIME'].max()
    print('source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max=\n',
          source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max)
    return source_iline_min, source_iline_max, source_xline_min, source_xline_max, time_min, time_max