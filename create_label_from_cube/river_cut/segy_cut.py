from label_location import label_location, label_location_read
from label_create import label_create
import segyio
def segy_cut(filename, label_file_list, slice_class='iline'):
    label_iline_start =label_xline_start = label_time_start = 99999
    label_iline_end = label_xline_end = label_time_end = 0
    df = []
    for m in range(len(label_file_list)):
        df.append(label_location_read(label_file_list[m]))
        label_iline_start_new, label_iline_end_new, label_xline_start_new, label_xline_end_new, label_time_start_new, label_time_end_new = \
            label_location(df[m])
        if label_iline_start_new < label_iline_start:
            label_iline_start = label_iline_start_new
        if label_xline_start_new < label_xline_start:
            label_xline_start = label_xline_start_new
        if label_time_start_new < label_time_start:
            label_time_start = label_time_start_new
        if label_iline_end_new > label_iline_end:
            label_iline_end = label_iline_end_new
        if label_xline_end_new > label_xline_end:
            label_xline_end = label_xline_end_new
        if label_time_end_new > label_time_end:
            label_time_end = label_time_end_new
    output_data = segyio.tools.cube(filename)    #origion seismic data
    output_label = segyio.tools.cube(filename)   #label data cube
    output_label[:, :, :] = 0
    data_iline_start, data_xline_start, data_time_start = get_location_start(filename)
    output_data = output_data[label_iline_start-data_iline_start:label_iline_end-data_iline_start+1, \
                  label_xline_start-data_xline_start:label_xline_end-data_xline_start+1, label_time_start-data_time_start:label_time_end-data_time_start+1]  # cut from segy file
    output_label = output_label[label_iline_start-data_iline_start:label_iline_end-data_iline_start+1, \
                  label_xline_start-data_xline_start:label_xline_end-data_xline_start+1, label_time_start-data_time_start:label_time_end-data_time_start+1] #slice from source data cube
    for m in range(len(label_file_list)):
        array_from_dataframe = df[m].to_numpy()
        for i in range(array_from_dataframe.shape[0]):
            inline = array_from_dataframe[i][0]
            xline = array_from_dataframe[i][1]
            time = array_from_dataframe[i][4]  # the time have been rouned at read_river
            output_label[inline - label_iline_start, xline - label_xline_start, time - label_time_start] = m+1
    if slice_class == 'iline' :
        label_create('iline', label_iline_start, label_iline_end, output_label, 'test')
    if slice_class == 'xline':
        label_create('xline', label_xline_start, label_xline_end, output_label, 'test')
    if slice_class == 'time':
        label_create('time', label_time_start, label_time_end, output_label, 'test')
    output_label = output_label.flatten()
    output_data = output_data.flatten()
    output_data.tofile("cut.bin")
    output_label.tofile('label.bin')
    return

def get_location_start(file):
    with segyio.open(file) as f:
        line_start = f.ilines[0]
        xline_start = f.xlines[0]
        time_start = int(f.samples[0])
    return line_start, xline_start, time_start