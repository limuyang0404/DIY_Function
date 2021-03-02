# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt
import math
def function_img(x_list, y_list):
    plt.plot(x_list, y_list, '.')
    axis = plt.gca()
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')
    axis.spines['left'].set_position(('data', 0))
    axis.spines['bottom'].set_position(('data', 0))
    plt.legend(['f(x)'])
    plt.show()
    return
x = np.linspace(-20 , 20, 401)
y = []
for i in x:
    y.append(math.exp(-1 * i) / ((1 + math.exp(-1 * i))**2))
# y = np.sin(x)
function_img(x, y)
