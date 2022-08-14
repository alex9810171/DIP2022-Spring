# generic libraries
import os
import glob
from cv2 import connectedComponents
import cv2

# customize libraries
import ocr

# configurations
config = {
    'doc_image_path': r'../data/processedImage/CamScanner/*.jpg',
    'test_doc_image_path': r'../data/processedImage/GoodScan/*.jpg',

    'marked_image_path': r'../data/processedImage/marked_letter.jpg',
    'extract_image_path': r'../data/processedImage/extracted_letter',
    'extract_img_format': '.png',

    'letter_image_path': r'../data/processedImage/extracted_letter/*.png',
    'letter_answer_path': r'../data/processedImage/answer_letter/*.jpg',

    'detect_all_letter': [0, 51],
    'detect_upper_letter': [26, 51],

    'space_size': 5,
    'newline_size': 15
}

if __name__=='__main__':
    # read document images
    images = [cv2.imread(file) for file in glob.glob(config['test_doc_image_path'])]
    answer_images = [cv2.imread(file) for file in glob.glob(config['letter_answer_path'])]
    
    # execute OCR API; letter_info = x, y, width, height, centroid_x, centroid_y
    marked_image, letter_images, letter_info, sentence = ocr.OCRAPI_for_Web(images, answer_images, config['space_size'], config['newline_size'], config['detect_upper_letter'])

    # output
    cv2.imwrite(config['marked_image_path'], marked_image)
    cv2.imshow('Marked_Image', marked_image)
    cv2.waitKey(0)

    for i in range(len(letter_info)):
        zero_str = ''
        zero_num = len(str(len(letter_info)))-len(str(i))
        for j in range(zero_num):
            zero_str = zero_str+'0'
        save_path_file = os.path.join(config['extract_image_path'], zero_str+str(i)+config['extract_img_format'])
        cv2.imwrite(save_path_file, letter_images[i])

    print(sentence)
