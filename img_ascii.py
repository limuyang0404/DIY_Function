# coding=UTF-8
import numpy as np
from PIL import Image
def rgba_to_ascii(img_path):
    img = Image.open(img_path)
    img_array = np.array(img)
    img_array_shape = img_array.shape
    print(img_array_shape)
    gray_value = np.zeros(img_array_shape)
    code = ""
    if img_array_shape[2] == 3:
        gray_value = img_array[:, :, 0] * 0.299 + img_array[:, :, 1] * 0.587 + img_array[:, :, 2] * 0.114
        # code = chr(int(gray_value))
        gray_value = (gray_value / 4).astype(int)
        print(gray_value)
        for i in range(img_array_shape[0]):
            for j in range(img_array_shape[1]):
                code += chr(gray_value[i, j] + 33)
            code += '\n'
        with open(r'C:\Users\Administrator\Desktop\1.txt', 'w') as f:
            f.write(code)
    return gray_value
if __name__ == '__main__':
    img = r'C:\Users\Administrator\Desktop\haha.png'
    gray = rgba_to_ascii(img)
