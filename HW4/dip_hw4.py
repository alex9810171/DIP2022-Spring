import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2

config = {
    'dither_matrix': np.array([[1, 2],
                               [3, 0]], dtype=int),
    'expand_size': 256,
    'Floyd_Steinberg_pattern': np.array([[0, 0, 7],
                                         [3, 5, 1]], dtype=int),
    'Floyd_Steinberg_param': 16,
    'Jarvis_pattern': np.array([[0, 0, 0, 7, 5],
                                [3, 5, 7, 5, 3],
                                [1, 3, 5, 3, 1]], dtype=int),
    'Jarvis_param': 48,
    'Atkinson_pattern': np.array([[0, 0, 0, 1, 1],
                                  [0, 1, 1, 1, 0],
                                  [0, 0, 1, 0, 0]], dtype=int),
    'Atkinson_param': 8,
    'resize_ratio': 0.5,
    'low_pass_filter_3x3': np.array([[1, 1, 1],
                                     [1, 2, 1],
                                     [1, 1, 1]], dtype=int),
    'low_pass_filter_3x3_param': 10
}

def expand_dither_matrix(dither_matrix_raw, expand_size):
    dither_matrix = dither_matrix_raw.copy()
    while(dither_matrix.shape[0] < expand_size):
        dither_matrix_expand = np.zeros((dither_matrix.shape[0]*2, dither_matrix.shape[1]*2), dtype=int)
        for j in range(0, dither_matrix_expand.shape[0], dither_matrix.shape[0]):
            for i in range(0, dither_matrix_expand.shape[1], dither_matrix.shape[0]):
                for j1 in range(dither_matrix.shape[0]):
                    for i1 in range(dither_matrix.shape[1]):
                        dither_matrix_expand[j+j1, i+i1] = 4*dither_matrix[j1, i1] + dither_matrix_raw[j//dither_matrix.shape[0], i//dither_matrix.shape[1]]
        dither_matrix = dither_matrix_expand.copy()
    return dither_matrix

def get_threshold_matrix(dither_matrix):
    threshold_matrix = np.zeros((dither_matrix.shape[0], dither_matrix.shape[1]), dtype=int)
    for j in range(dither_matrix.shape[0]):
        for i in range(dither_matrix.shape[1]):
            threshold_matrix[j, i] = int(255*(dither_matrix[j, i]+0.5)//dither_matrix.shape[0]**2)
    return threshold_matrix

def dither(img, dither_matrix):
    img_dither = Image.new('1', (img.width, img.height))
    threshold_matrix = get_threshold_matrix(dither_matrix)
    for y in range(0, img.height, threshold_matrix.shape[0]):
        for x in range(0, img.width, threshold_matrix.shape[1]):
            for j in range(threshold_matrix.shape[0]):
                for i in range(threshold_matrix.shape[1]):
                   if(img.getpixel((x+i, y+j)) >= threshold_matrix[j, i]):
                       img_dither.putpixel((x+i, y+j), 255)
    return img_dither

def error_diffuse(img, threshold, pattern, pattern_param):
    img_dither = Image.new('1', (img.width, img.height))
    array_img = np.array(img, dtype=float)
    array_img = np.pad(array_img, ((0, pattern.shape[0]-1), (pattern.shape[1]//2, pattern.shape[1]//2)), 'edge')
    for j in range(array_img.shape[0]-pattern.shape[0]+1):
        for i in range(pattern.shape[1]//2, array_img.shape[1]-pattern.shape[1]//2):
            norm = array_img[j, i]/255
            g = 0
            if(norm >= threshold):
                g = 1
                img_dither.putpixel((i-pattern.shape[1]//2, j), 255)
            error = norm-g
            for j1 in range(pattern.shape[0]):
                for i1 in range(pattern.shape[1]):
                    array_img[j+j1, i+i1-pattern.shape[1]//2] += error*pattern[j1, i1]/pattern_param*255
    return img_dither

def filt(img, filter, filter_param):
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((filter.shape[0]//2, filter.shape[0]//2), (filter.shape[1]//2, filter.shape[1]//2)), 'edge')
    img_new = Image.new('L', (img.width, img.height))
    for y in range(filter.shape[0]//2, array_img.shape[0]-filter.shape[0]//2):
        for x in range(filter.shape[1]//2, array_img.shape[1]-filter.shape[1]//2):
            sum = 0
            for j in range(filter.shape[0]):
                for i in range(filter.shape[1]):
                    sum += array_img[y+j-filter.shape[0]//2, x+i-filter.shape[1]//2]*filter[j, i]
            img_new.putpixel((x-filter.shape[1]//2, y-filter.shape[0]//2), int(sum/filter_param))
    return img_new
'''
def unsharp_masking(array_img, c):      # 0.6 <= c <= 0.83333
    array_new = np.zeros((array_img.shape[0], array_img.shape[1]), dtype=int)
    low_pass_filter_3x3 = np.array([[1, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 1]], dtype=int)
    array_img = np.pad(array_img, ((1, 1), (1, 1)), 'edge')
    img_l = Image.new('L', (img.width, img.height))
    for y in range(1, array_img.shape[0]-1):
        for x in range(1, array_img.shape[1]-1):
            sum = 0
            for j in range(0, 3):
                for i in range(0, 3):
                    sum += array_img[y+j-1, x+i-1]*low_pass_filter_3x3[j, i]
            img_l.putpixel((x-1, y-1), int(sum/10))
    for y in range(img_g.height):
        for x in range(img_g.width):
            g = c/(2*c-1)*img.getpixel((x, y))-(1-c)/(2*c-1)*img_l.getpixel((x, y))
            img_g.putpixel((x, y), int(g))
    return array_new
'''

if __name__=='__main__':
    # create result folder & read image
    os.makedirs('ResultImage', exist_ok=True)
    sample1 = Image.open("SampleImage/sample1.png")
    sample2 = Image.open("SampleImage/sample2.png")
    sample2_cv2 = cv2.imread("SampleImage/sample2.png")
    sample3 = Image.open("SampleImage/sample3.png")
    sample3_cv2 = cv2.imread("SampleImage/sample3.png")
    
    result1 = dither(sample1, config['dither_matrix'])
    result1.save("ResultImage/result1.png")

    dither_matrix_expand = expand_dither_matrix(config['dither_matrix'], config['expand_size'])
    result2 = dither(sample1, dither_matrix_expand)
    result2.save("ResultImage/result2.png")

    result3 = error_diffuse(sample1, 0.5, config['Floyd_Steinberg_pattern'], config['Floyd_Steinberg_param'])
    result3.save("ResultImage/result3.png")

    result4 = error_diffuse(sample1, 0.5, config['Jarvis_pattern'], config['Jarvis_param'])
    result4.save("ResultImage/result4.png")
    
    # reference1 = error_diffuse(sample1, 0.5, config['Atkinson_pattern'], config['Atkinson_param'])
    # reference1.save("Reference/reference1.png")

    result5 = cv2.resize(sample2_cv2, None, fx=config['resize_ratio'], fy=config['resize_ratio'])
    cv2.imwrite("ResultImage/result5.png", result5)

    # sample2_low_pass = filt(sample2, config['low_pass_filter_3x3'], config['low_pass_filter_3x3_param'])
    # sample2_low_pass_cv2 = cv2.cvtColor(np.array(sample2_low_pass), cv2.COLOR_RGB2BGR)
    # reference2 = cv2.resize(sample2_low_pass_cv2, None, fx=config['resize_ratio'], fy=config['resize_ratio'])
    # cv2.imwrite("Reference/reference2.png", reference2)

    #np.fft.fft2(sample3_cv2)
