import cv2
import numpy as np


def adjust_contrast_brightness(img, alpha=1.0, beta=0.0):
    return cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)


def gamma_correction(img, gamma=1.0, gain=1.0):
    img = img.astype(np.float32) / 255.0
    img = img * gain
    img = img ** gamma
    img = img * 255
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def simplest_cb(img, percent=1):
    """
    Apply color balance to an image.
    Ref: http://web.stanford.edu/~sujason/ColorBalancing/simplestcb.html
    Ref2: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc
    """
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist(
            [channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


def top_hat_transform(img, kernel_size=21):
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (kernel_size, kernel_size))
    invert = 255-img
    return (255 - cv2.morphologyEx(invert, cv2.MORPH_TOPHAT, kernel)).astype(np.uint8)
