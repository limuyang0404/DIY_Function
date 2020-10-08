import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

def img_filename_class(filename_in, filename_out):
    """This function is designed to color the img in minist to classify them"""
    class_color_coding = [
        [156, 0, 147],  # blue
        [237, 29, 37],  # red
        [0, 255, 0],  # green
        [0, 255, 255],  # cyan
        [255, 0, 255],  # blue
        [255, 255, 0],
        [255, 246, 143],
        [255, 193, 193],
        [25, 106, 106],
        [99, 184, 255]  # yellow
    ]
    #class_color_coding is the list used coloring those imgs
    files = [f for f in listdir(filename_in) if
             isfile(join(filename_in, f))]
    m = 0
    for file in files:
        a = int(file[0])
        img = plt.imread(join(filename_in, file))
        img1 = img*1
        img2 = img*1
        img3 = img*1
        img_row = img1.shape[0]
        img_col = img1.shape[1]
        for i in range(img_row):
            for j in range(img_col):
                if img1[i][j]>10:
                    img1[i][j] = class_color_coding[a][0]
                    img2[i][j] = class_color_coding[a][1]
                    img3[i][j] = class_color_coding[a][2]
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
        img3 = np.expand_dims(img3, axis=2)
        img_rgb = np.concatenate((img1, img2, img3), axis=2)
        im = Image.fromarray(img_rgb)
        im = im.convert('RGB')
        im.save(join(filename_out, file))
        m += 1
        print('the', m, 'th img have been classified')

    return
# img_filename_class(r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\mnist_train_jpg_60000', r'D:\mnist-unet\mnist_data_jpg\mnist_data_jpg\mnist_train_jpg_60000_classify')




