import os
import numpy as np
import cv2

config = {
    'kernel': np.ones((3,3), np.uint8),
    'save_path': r'../data/processedImage/',
    'preprocess_image_name': 'erosion',
    'preprocess_image_format': '.jpg',
    'show': True
}

def cvtColor_BGR2BINARYINV(src):
    gray_img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    return binary_img

def preprocess(src):
    erosion_img = cv2.morphologyEx(src, cv2.MORPH_OPEN, config['kernel'])
    save_path_file = os.path.join(config['save_path'], config['preprocess_image_name']+config['preprocess_image_format'])
    cv2.imwrite(save_path_file, erosion_img)
    if(config['show']):
        cv2.imshow('Result', erosion_img)
        cv2.waitKey(0)
    return erosion_img
