from PIL import Image
import numpy as np
import random
def cube_creater(path, output_path, cube_number):
    color = [[255, 0, 0],
             [255, 255, 0],
             [0, 255, 255]]
    cube_raw = Image.open(path)
    cube_raw = np.asarray(cube_raw)
    cube_output = cube_raw*1
    for i in range(cube_number):
        random_r1 = random.randint(0, 255)
        random_g1 = random.randint(0, 255)
        random_b1 = random.randint(0, 255)
        random_color1 = [random_r1, random_g1, random_b1]
        random_r2 = random.randint(0, 255)
        random_g2 = random.randint(0, 255)
        random_b2 = random.randint(0, 255)
        random_color2 = [random_r2, random_g2, random_b2]
        random_r3 = random.randint(0, 255)
        random_g3 = random.randint(0, 255)
        random_b3 = random.randint(0, 255)
        random_color3 = [random_r3, random_g3, random_b3]
        for j in range(cube_raw.shape[0]):
            for k in range(cube_raw.shape[1]):
                if list(cube_raw[j, k, 0:3])==color[0]:
                    cube_output[j, k, 0:3] = random_color1
                if list(cube_raw[j, k, 0:3])==color[1]:
                    cube_output[j, k, 0:3] = random_color2
                if list(cube_raw[j, k, 0:3])==color[2]:
                    cube_output[j, k, 0:3] = random_color3
        im = Image.fromarray(cube_output)
        print('The '+str(i+1)+'th cube is being created')
        im = im.convert('RGBA')
        im.save(output_path+r'\cube'+str(i+1)+'.png')
    print('All cube('+str(i+1)+') have been created!')
    return
# cube_creater(r'C:\Users\Administrator\Desktop\ppt制作\cube.png', r'C:\Users\Administrator\Desktop\ppt制作\random_cube', 200)



