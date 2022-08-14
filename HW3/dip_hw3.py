from email.policy import default
import os
from pickle import FALSE
import sys
import math
from turtle import distance
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# PIL '1' mode will operate in 8-bit, 0/255 type.
# Therefore, be carefull when doing 0/255 or 0/1 operations.

SAMPLE1_WIDTH = int(600)
SAMPLE1_HEIGHT = int(600)

sys.setrecursionlimit(SAMPLE1_WIDTH*SAMPLE1_HEIGHT)

all_1_kernel_3x3 = np.array([[1, 1, 1],
                             [1, 1, 1],
                             [1, 1, 1]], dtype=int)

g0_array = np.array([[105, 245],
                     [135, 235],
                     [125, 281],
                     [207, 420],
                     [268, 407],
                     [239, 500],
                     [305, 472],
                     [302, 490],
                     [295, 455]], dtype=int)

img_hole_filling = Image.new('1', (SAMPLE1_WIDTH, SAMPLE1_HEIGHT))    

kernel_law2_3x3 = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=int)     # 1/12

feature_vector = np.array([[40*15, 25*15],
                          [15*15, 25*15],
                          [32*15, 36*15],
                          [35*15, 13*15],
                          [27*15, 11*15]], dtype=int)

'''feature_vector = np.array([[511, 257],
                            [459, 149],
                            [507, 213],
                            [471, 488],
                            [237, 152]], dtype=int)'''

feature_vector_new = np.array([[40*15, 25*15],
                          [35*15, 13*15],
                          [27*15, 11*15]], dtype=int)

def erode(img, kernel, iterations=1):
    img_erode = Image.new('1', (img.width, img.height))
    for y in range(1, img.height-1):
        for x in range(1, img.width-1):
            detected = False
            for j in range(kernel.shape[0]):
                for i in range(kernel.shape[1]):
                    if(img.getpixel((x+i-kernel.shape[0]//2, y+j-kernel.shape[1]//2))//255 != kernel[j, i]):
                        detected = True
                        i = kernel.shape[1]-1
                        j = kernel.shape[0]-1
            if (not detected):
                img_erode.putpixel((x, y), 1*255)
    if(iterations > 1):
        return erode(img_erode, kernel, iterations-1)
    else:
        return img_erode

def extract_img_boundary(img):
    img_boundary = Image.new('1', (img.width, img.height))
    img_erosion = erode(img, all_1_kernel_3x3)
    for y in range(1, img.height-1):
        for x in range(1, img.width-1):
            pixel = img.getpixel((x, y))-img_erosion.getpixel((x, y))
            if(pixel < 0):
                pixel = 255
            img_boundary.putpixel((x, y), pixel//255)
    return img_boundary

def flood_fill(x, y, color, img):
    img_hole_filling.putpixel((x, y), color)
    if(x > 0 and img.getpixel((x-1, y)) == 0 and img_hole_filling.getpixel((x-1, y)) == 0):
        flood_fill(x-1, y, color, img)
    if(y > 0 and img.getpixel((x, y-1)) == 0 and img_hole_filling.getpixel((x, y-1)) == 0):
        flood_fill(x, y-1, color, img)
    if(x < SAMPLE1_WIDTH and img.getpixel((x+1, y)) == 0 and img_hole_filling.getpixel((x+1, y)) == 0):
        flood_fill(x+1, y, color, img)
    if(y < SAMPLE1_HEIGHT and img.getpixel((x, y+1)) == 0 and img_hole_filling.getpixel((x, y+1)) == 0):
        flood_fill(x, y+1, color, img)

def fill_hole(img, g0):
    img_new = img.copy()
    for g in range(g0.shape[0]):
        flood_fill(g0[g, 0], g0[g, 1], 1*255, img)
    for y in range(img.height):
        for x in range(img.width):
            if(img_hole_filling.getpixel((x, y)) == 1*255):
                img_new.putpixel((x, y), 1*255)
    return img_new

'''
def skeletonize(img, iterations):
    img_new = img.copy()
    detected = False
    for y in range(1, img.height-1):
        for x in range(1, img.width-1):
            if(img.getpixel((x, y)) == 1*255):
                count = 0
                for j in range(3):
                    for i in range(3):
                        if(img.getpixel((x+i-1, y+j-1)) == 0):
                            count +=1
                if(count > 0 and count < 5):
                    img_new.putpixel((x, y), 0)
                    detected = True
    if(not detected):
        iterations = 1
    if(iterations > 1):
        return skeletonize(img_new, iterations-1)
    else:
        return img_new
'''

def count_yokoi(pos_x, pos_y, array_binary):
    q = 0
    r = 0
    # top-right
    if(pos_x+1 < array_binary.shape[1] and pos_y-1 >= 0 and array_binary[pos_y, pos_x+1] == 1
    and array_binary[pos_y-1, pos_x] == 1 and array_binary[pos_y-1, pos_x+1] == 1):
        r += 1
    elif(pos_x+1 < array_binary.shape[1] and array_binary[pos_y, pos_x+1] == 1):
        q += 1
    # top-left
    if(pos_x-1 >= 0 and pos_y-1 >= 0 and array_binary[pos_y-1, pos_x] == 1
    and array_binary[pos_y, pos_x-1] == 1 and array_binary[pos_y-1, pos_x-1] == 1):
        r += 1
    elif(pos_y-1 >= 0 and array_binary[pos_y-1, pos_x] == 1):
        q += 1
    # bottom-left
    if(pos_x-1 >= 0 and pos_y+1 < array_binary.shape[0] and array_binary[pos_y, pos_x-1] == 1
    and array_binary[pos_y+1, pos_x] == 1 and array_binary[pos_y+1, pos_x-1] == 1):
        r += 1
    elif(pos_x-1 >= 0 and array_binary[pos_y, pos_x-1] == 1):
        q += 1
    # bottom-right
    if(pos_x+1 < array_binary.shape[1] and pos_y+1 < array_binary.shape[0]
    and array_binary[pos_y+1, pos_x] == 1 and array_binary[pos_y, pos_x+1] == 1
    and array_binary[pos_y+1, pos_x+1] == 1):
        r += 1
    elif(pos_y+1 < array_binary.shape[0] and array_binary[pos_y+1, pos_x] == 1):
        q += 1
    if(r == 4):
        yokoi_number = 5
    else:
        yokoi_number = q
    return yokoi_number

def skeletonize(img):
    # transform to binary array
    array_binary_new = np.zeros((img.height, img.width), dtype=int)
    for y in range(img.height):
        for x in range(img.width):
            array_binary_new[y, x] = img.getpixel((x, y))//255
    
    # skeletonize
    equal = False
    while equal == False:
        array_binary_old = array_binary_new.copy()
        array_yokoi_matrix = array_binary_old.copy()
        # step 1: count Yokoi connectivity number
        for j in range(array_binary_old.shape[0]):
            for i in range(array_binary_old.shape[1]):
                if(array_binary_old[j, i] == 1):
                    array_yokoi_matrix[j, i] = count_yokoi(i, j, array_binary_old)
        # step 2: generate pair relationship matrix, for each pixel: p = 1, q = 2, background = 0
        array_pair_relationship_matrix = np.zeros((array_yokoi_matrix.shape[0], array_yokoi_matrix.shape[1]), dtype=int)
        for j in range(array_yokoi_matrix.shape[0]):
            for i in range(array_yokoi_matrix.shape[1]):
                # check if self is edge and have an edge neighbor
                if(array_yokoi_matrix[j, i] == 1):
                    if((i+1 < array_yokoi_matrix.shape[1] and array_yokoi_matrix[j, i+1] == 1)
                    or (j-1 >= 0 and array_yokoi_matrix[j-1, i] == 1)
                    or (i-1 >= 0 and array_yokoi_matrix[j, i-1] == 1)
                    or (j+1 < array_yokoi_matrix.shape[0] and array_yokoi_matrix[j+1, i] == 1)):
                        array_pair_relationship_matrix[j, i] = 1
                    else:
                        array_pair_relationship_matrix[j, i] = 2
        # step 3: if pair relationship operator is p, then check if connected shrink operator is removable (yokoi = 1)
        #         finally, delete those pixels satisfied above two conditions 
        for j in range(array_binary_new.shape[0]):
            for i in range(array_binary_new.shape[1]):
                if(array_pair_relationship_matrix[j, i] == 1):
                    if(count_yokoi(i, j, array_binary_new) == 1):
                        array_binary_new[j, i] = 0
        equal = np.array_equal(array_binary_old, array_binary_new)
    
    # array to img
    img_skeleton = Image.new('1', (img.width, img.height))
    for y in range(array_binary_new.shape[0]):
        for x in range(array_binary_new.shape[1]):
            img_skeleton.putpixel((x, y), int(array_binary_new[y, x]*255))
    return img_skeleton

def reverse_binary_image(img):
    img_reverse = Image.new('1', (img.width, img.height))
    for y in range(img.height):
        for x in range(img.width):
            if(img.getpixel((x, y)) == 0):
                img_reverse.putpixel((x, y), 255)
    return img_reverse 

def connected_component(img):
    # pro-process image to array
    im_connected = np.array(img, dtype=int)

    # iterative algorithm to get component
    # initialize each pixel to a unique label
    label_count = 1
    for j in range(im_connected.shape[0]):
        for i in range(im_connected.shape[1]):
            if(im_connected[j, i] == 1):
                im_connected[j, i] = label_count
                label_count += 1

    change = True
    while change == True:
        # top-down pass
        change = False
        for j in range(im_connected.shape[0]):
            for i in range(im_connected.shape[1]):
                # find a component
                if(im_connected[j, i] > 0):
                    # first row
                    if(j == 0):
                        # neighbor is a component
                        if(i > 0 and im_connected[j, i-1] > 0 and im_connected[j, i-1] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j, i-1]
                            change = True
                    # second row or below
                    else:
                        # neighbor exists at left and top
                        if(i > 0 and im_connected[j, i-1] > 0 and im_connected[j-1, i] > 0):
                            # choose a smaller component to override
                            if(im_connected[j, i-1] <= im_connected[j-1, i] and im_connected[j, i-1] < im_connected[j, i]):
                                im_connected[j, i] = im_connected[j, i-1]
                                change = True
                            elif(im_connected[j, i-1] > im_connected[j-1, i] and im_connected[j-1, i] < im_connected[j, i]):
                                im_connected[j, i] = im_connected[j-1, i]
                                change = True
                        # neighbor exists at left
                        elif(i > 0 and im_connected[j, i-1] > 0 and im_connected[j, i-1] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j, i-1]
                            change = True
                        # neighbor exists at top
                        elif(im_connected[j-1, i] > 0 and im_connected[j-1, i] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j-1, i]
                            change = True
        # bottom-up pass
        for j in range(im_connected.shape[0]-1, -1, -1):
            for i in range(im_connected.shape[1]-1, -1, -1):
                # find a component
                if(im_connected[j, i] > 0):
                    # bottom row
                    if(j == im_connected.shape[0]-1):
                        # neighbor is a component
                        if(i < im_connected.shape[1]-1 and im_connected[j, i+1] > 0 and im_connected[j, i+1] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j, i+1]
                            change = True
                    # upon bottom row
                    else:
                        # neighbor exists at right and bottom
                        if(i < im_connected.shape[1]-1 and im_connected[j, i+1] > 0 and im_connected[j+1, i] > 0):
                            # choose a smaller component to override
                            if(im_connected[j, i+1] <= im_connected[j+1, i] and im_connected[j, i+1] < im_connected[j, i]):
                                im_connected[j, i] = im_connected[j, i+1]
                                change = True
                            elif(im_connected[j, i+1] > im_connected[j+1, i] and im_connected[j+1, i] < im_connected[j, i]):
                                im_connected[j, i] = im_connected[j+1, i]
                                change = True
                        # neighbor exists at right
                        elif(i < im_connected.shape[1]-1 and im_connected[j, i+1] > 0 and im_connected[j, i+1] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j, i+1]
                            change = True
                        # neighbor exists at bottom
                        elif(im_connected[j+1, i] > 0 and im_connected[j+1, i] < im_connected[j, i]):
                            im_connected[j, i] = im_connected[j+1, i]
                            change = True
    
    # output RGB image
    img_new = Image.new('RGB', (img.width, img.height))
    for y in range(img_new.height):
        for x in range(img_new.width):
            if(im_connected[y, x] > 0):
                r = im_connected[y, x]//1000*2+30
                g = im_connected[y, x]%1000//100*20+30
                b = im_connected[y, x]%100*2+30
                img_new.putpixel((x, y), (r, g, b))
    return img_new

def laws_method(img, kernel, window_size, write_txt=False):
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((1, 1), (1, 1)), 'edge')
    array_microstructure = np.zeros((img.height, img.width), dtype=int)
    array_energy = np.zeros((img.height//window_size[1], img.width//window_size[0]), dtype=int)
    for y in range(1, array_img.shape[0]-1):
        for x in range(1, array_img.shape[1]-1):
            sum = 0
            for j in range(kernel.shape[0]):
                for i in range(kernel.shape[1]):
                    sum += array_img[y+j-1, x+i-1]*kernel[j, i]
            sum /= 12
            array_microstructure[y-1, x-1] = sum
    for y in range(0, array_microstructure.shape[0], window_size[1]):
        for x in range(0, array_microstructure.shape[1], window_size[0]):
            energy = 0
            for j in range(window_size[1]):
                for i in range(window_size[0]):
                    energy += array_microstructure[y+j, x+i]
            array_energy[y//window_size[1], x//window_size[0]] = energy
    if(write_txt):
        min = 100000
        max = -100000
        with open('Reference/sample2_energy.txt', 'w') as file:
            for y in range(array_energy.shape[0]):
                for x in range(array_energy.shape[1]):
                    file.write('%d ' %(array_energy[y, x]))
                    if(array_energy[y, x] > max):
                        max = array_energy[y, x]
                    if(array_energy[y, x] < min):
                        min = array_energy[y, x]
                file.write('\n')
        img_new = Image.new('L', (array_energy.shape[1], array_energy.shape[0]))
        for y in range(array_energy.shape[0]):
            for x in range(array_energy.shape[1]):
                img_new.putpixel((x, y), int((array_energy[y,x]-min)/(max-min)*255))
        img_new.save("Reference/sample2_energy.png")
        plt.imshow(img_new)
        for i in range(feature_vector.shape[0]):
            plt.plot(feature_vector[i, 0], feature_vector[i, 1], 'r*')
        plt.show()
    return array_energy

def k_means_on_image(img, array_feature, iterations=20):
    img_new = Image.new('L', (img.width, img.height))
    count = 0
    equal = False
    while(not equal):
        array_feature_old = array_feature.copy()
        k_min = {}
        for k in range(array_feature.shape[0]):
            k_min[k] = []
        for y in range(img.height):
            for x in range(img.width):
                min_distance = 1000
                min_k = 0
                for k in range(array_feature.shape[0]):
                    # distance = ((x-array_feature[k, 0])**2+(y-array_feature[k, 1])**2)**(1/2)
                    distance = abs(img.getpixel((array_feature[k, 0], array_feature[k, 1])) - img.getpixel((x, y)))
                    if(distance < min_distance):
                        min_distance = distance
                        min_k = k
                k_min[min_k].append([x, y, distance])
        for i in range(array_feature.shape[0]):
            if(len(k_min[i]) != 0):
                centroid_x = 0
                centroid_y = 0
                for j in range(len(k_min[i])):
                    x = k_min[i][j][0]
                    y = k_min[i][j][1]
                    centroid_x += x
                    centroid_y += y
                array_feature[i, 0] = centroid_x // len(k_min[i])
                array_feature[i, 1] = centroid_y // len(k_min[i])
        equal = np.array_equal(array_feature, array_feature_old)
        count += 1
        if(count >= iterations):
            equal = True
        if(equal):
            for i in range(array_feature.shape[0]):
                for j in range(len(k_min[i])):
                    x = k_min[i][j][0]
                    y = k_min[i][j][1]
                    img_new.putpixel((x, y), i*50)
    return img_new

if __name__=='__main__':
    # create result folder & read image
    os.makedirs('ResultImage', exist_ok=True)
    sample1 = Image.open("SampleImage/sample1.png").convert('1')    # convert to b/w
    sample2 = Image.open("SampleImage/sample2.png")
    sample3 = Image.open("SampleImage/sample3.png")

    result1 = extract_img_boundary(sample1)
    result1.save("ResultImage/result1.png")

    result2 = fill_hole(sample1, g0_array)
    result2.save("ResultImage/result2.png")

    result3 = skeletonize(sample1)
    result3.save("ResultImage/result3.png")
    
    sample1_reverse = reverse_binary_image(sample1)
    result4 = skeletonize(sample1_reverse)
    result4.save("ResultImage/result4.png")

    result5 = connected_component(sample1)
    result5.save("ResultImage/result5.png")

    #laws_method(sample2, kernel_law2_3x3, np.array([15, 15]), write_txt=True)

    result6 = k_means_on_image(sample2, feature_vector)
    result6.save("ResultImage/result6.png")

    result7 = k_means_on_image(sample2, feature_vector_new)
    result7.save("ResultImage/result7.png")
