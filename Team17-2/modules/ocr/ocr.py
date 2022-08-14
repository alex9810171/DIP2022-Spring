import cv2
import modules.ocr.preprocess_img as preprocess_img
# import preprocess_img

config = {
    'component_size': [10, 400],
    'tolerant_pos': 15
}

dic_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j',
              10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't',
              20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z', 26: 'A', 27: 'B', 28: 'C', 29: 'D',
              30: 'E', 31: 'F', 32: 'G', 33: 'H', 34: 'I', 35: 'J', 36: 'K', 37: 'L', 38: 'M', 39: 'N',
              40: 'O', 41: 'P', 42: 'Q', 43: 'R', 44: 'S', 45: 'T', 46: 'U', 47: 'V', 48: 'W', 49: 'X',
              50: 'Y', 51: 'Z', -1: ' ERROR! '}


def extract_letter(image):
    marked_image = image.copy()
    binary_inv_image = preprocess_img.cvtColor_BGR2BINARYINV(image)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary_inv_image)
    letter_image = []
    letter_info = []

    # traverse all components
    for i in range(len(stats)):
        # if components fit the size, then mark letter and extract into each image
        size = stats[i][4]
        if(size > config['component_size'][0] and size < config['component_size'][1]):
            x1 = stats[i][0]
            y1 = stats[i][1]
            x2 = stats[i][0]+stats[i][2]
            y2 = stats[i][1]+stats[i][3]
            cv2.rectangle(marked_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            letter_image.append(image[y1:y2, x1:x2])
            info = [stats[i][0], stats[i][1], stats[i][2], stats[i]
                    [3], stats[i][4], centroids[i][0], centroids[i][1]]
            letter_info.append(info)

    # correct letter position
    for i in range(len(letter_info)):
        letter_x = letter_info[i][0]
        letter_y = letter_info[i][1]
        letter_info[i][0] = letter_x
        letter_info[i][1] = letter_y//config['tolerant_pos'] * \
            config['tolerant_pos']

    # sort letter by y-axis
    for i in range(len(letter_info)-1):
        for j in range(len(letter_info)-1-i):
            if(letter_info[j][1] > letter_info[j+1][1]):
                letter_image[j], letter_image[j +
                                              1] = letter_image[j+1], letter_image[j]
                letter_info[j], letter_info[j +
                                            1] = letter_info[j+1], letter_info[j]

    # sort letter by x-axis while y-axis is the same
    for i in range(len(letter_info)-1):
        for j in range(len(letter_info)-1-i):
            if(letter_info[j][0] > letter_info[j+1][0] and letter_info[j][1] == letter_info[j+1][1]):
                letter_image[j], letter_image[j +
                                              1] = letter_image[j+1], letter_image[j]
                letter_info[j], letter_info[j +
                                            1] = letter_info[j+1], letter_info[j]

    return marked_image, letter_image, letter_info


def get_similarity(pic1, pic2):
    similarity = 0
    for i in range(pic2.shape[0]):
        for j in range(pic2.shape[1]):
            if(pic1[i, j] == pic2[i, j]):
                similarity += 1
    return similarity


def pixelwise_recognition(letter, answer, detect_range):
    max_similarity = 0
    ans = 0
    for i in range(detect_range[0], detect_range[1]+1):
        similarity = get_similarity(letter, answer[i])
        if(similarity > max_similarity):
            max_similarity = similarity
            ans = i
    return dic_letter.get(ans)


def OCRAPI_for_Web(images, answer_images, space_size, newline_size, detect_mode):
    # extract letter and save letter image
    marked_image, letter_images, letter_info = extract_letter(images[0])

    # read letter images & convert to binary images
    letter_images = [cv2.resize(img, (10, 10)) for img in letter_images]
    binary_inv_letter_images = [
        preprocess_img.cvtColor_BGR2BINARYINV(img) for img in letter_images]

    answer_images = [cv2.resize(img, (10, 10)) for img in answer_images]
    binary_inv_answer_images = [
        preprocess_img.cvtColor_BGR2BINARYINV(img) for img in answer_images]

    # recognize image and ouput
    sentence = ''
    for i in range(len(binary_inv_letter_images)):
        letter = pixelwise_recognition(
            binary_inv_letter_images[i], binary_inv_answer_images, detect_mode)
        if(i > 0 and letter_info[i][1]-letter_info[i-1][1] > newline_size):
            sentence = sentence+'\n'+letter
        elif(i > 0 and letter_info[i][0]-(letter_info[i-1][0]+letter_info[i-1][2]) > space_size):
            sentence = sentence+' '+letter
        else:
            sentence = sentence+letter

    return marked_image, letter_images, letter_info, sentence
