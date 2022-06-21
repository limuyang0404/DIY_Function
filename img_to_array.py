# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.colors as colors
def img_to_array(img_path, class_color_list):
    a = image.imread(img_path)
    print('a.shape:', a.shape)
    a = (a*255).astype(int)
    b = np.zeros(shape=(a.shape[0], a.shape[1]), dtype='float32')
    for i in range(len(class_color_list)):
        b[np.where((a[:, :, 0]==class_color_list[i][0]) & (a[:, :, 1]==class_color_list[i][1]) & (a[:, :, 2]==class_color_list[i][2]))] = i
    class_color_list_new = [([y / 255 for y in x]) for x in class_color_list]
    bounds = np.arange(len(class_color_list) +1)
    bounds = [x - 0.5 for x in bounds]
    ticks = np.arange(len(class_color_list))
    color_bar = class_color_list_new
    cmap = colors.ListedColormap(color_bar)
    norms = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(b, cmap=cmap, aspect='auto', norm=norms)
    cbar = plt.colorbar()
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks)
    plt.show()
    return b
def lith_to_r(lith_array, velocity_list, density_list):
    velocity_cube = np.zeros(shape=lith_array.shape, dtype='float32')
    density_cube = np.zeros(shape=lith_array.shape, dtype='float32')
    for i in range(len(velocity_list)):
        index_cube = np.where(lith_array==i)
        velocity_cube[index_cube] = velocity_list[i]
        density_cube[index_cube] = density_list[i]
    plt.subplot(1, 3, 1)
    plt.imshow(lith_array, cmap=plt.cm.rainbow, aspect='auto')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(velocity_cube, cmap=plt.cm.rainbow, aspect='auto')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(density_cube, cmap=plt.cm.rainbow, aspect='auto')
    plt.colorbar()
    plt.show()
    return
def gaussian_fold(x_value, z_value, bk_list, ck_list, deviation_list):
    output_value = 0
    for i in range(len(bk_list)):
        output_value += bk_list[i] * np.exp(-1*(x_value-ck_list[i])**2/(2*deviation_list[i]**2))
    output_value = output_value * z_value * 1.5 / 400
    return output_value
def fold_r_model(model_shape, model_parameter_list):
    r_cube = np.zeros(shape=model_shape, dtype='float32')
    output_cube = np.zeros(shape=model_shape, dtype='float32')
    x_list = []
    y_list = []
    r_list = []
    i = 0
    while i < r_cube.shape[-1]:
    # for i in range(output_cube.shape[-1]):
        r_cube[:, i] = np.random.uniform(-1, 1)
        new_index = np.random.randint(5, 15)
        i += new_index
    for i in range(r_cube.shape[0]):
        for j in range(r_cube.shape[1]):
            x_list.append(i)
            y_list.append(j)
            r_list.append(r_cube[i, j])
    # y_list_shift = [model_parameter_list[0] * (x - int(model_shape[0] / 2)) for x in x_list]#dip
    # y_list_shift2 = [gaussian_fold(x_list[i], y_list[i], model_parameter_list[1], model_parameter_list[2], model_parameter_list[3]) for i in range(len(y_list))]
    # y_list1 = [y_list_shift[i] + y_list[i] for i in range(len(y_list))]
    y_list_shift2 = [gaussian_fold(x_list[i], y_list[i], model_parameter_list[1], model_parameter_list[2], model_parameter_list[3]) for i in range(len(y_list))]
    print('1', y_list[:20])
    print('2', y_list_shift2[:20])
    y_list2 = [y_list_shift2[i] + y_list[i] for i in range(len(y_list))]
    print('3', y_list2[:20])
    y_list3 = [int(y) for y in y_list2]
    for i in range(len(x_list)):
        if 0<=x_list[i]<model_shape[0] and 0 <= y_list3[i] < model_shape[1]:
            output_cube[x_list[i], y_list3[i]] = r_list[i]
    plt.imshow(np.moveaxis(output_cube, 0, 1), cmap=plt.cm.rainbow, aspect='auto')
    plt.colorbar()
    plt.show()
    return output_cube
if __name__ == "__main__":
    print('hello')
    # lith_array = img_to_array(r"model1.png", [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0)])
    # lith_to_r(lith_array=lith_array, velocity_list=[100, 300, 90, 220, 1300, 30, 3700], density_list=[0.3, 0.7, 3.1, 2.0, 1.1, 0.5, 0.1])
    fold_r_model((300, 400), [0.1, [40, -40, 25, -25, 10, -10], [130, 140, 150, 160, 170, 150], [30, 32, 34, 36, 38, 40]])
    # a1 = [0, 3, 6, 9]
    # a2 = [1, 4, 7, 10]
    # a3 = [0.1, 0.2, 0.7, -0.3, 1.3]
    # b = [(a1[i] + a2[i]) for i in np.arange(len(a1))]
    # print(b)


