import cv2
import numpy as np


def bradley_roth_threshold(img, w_size=3, w=40 / 255):
    # Obtaining rows and cols
    rows, cols = img.shape
    i_rows, i_cols = rows + 1, cols + 1
    w_size = int(cols / 18)
    # Computing integral image
    # Leaving first row and column in zero for convenience
    integ = np.zeros((i_rows, i_cols), np.float32)

    integ[1:, 1:] = np.cumsum(
        np.cumsum(img.astype(np.float32), axis=0), axis=1)

    # Defining grid
    x, y = np.meshgrid(np.arange(1, i_cols), np.arange(1, i_rows))

    # Obtaining local coordinates
    hw_size = w_size // 2
    x1 = (x - hw_size).clip(1, cols)
    x2 = (x + hw_size).clip(1, cols)
    y1 = (y - hw_size).clip(1, rows)
    y2 = (y + hw_size).clip(1, rows)

    # Obtaining local areas size
    l_size = (y2 - y1 + 1) * (x2 - x1 + 1)

    # Computing sums
    sums = (integ[y2, x2] - integ[y2, x1 - 1] -
            integ[y1 - 1, x2] + integ[y1 - 1, x1 - 1])

    # Computing local means
    means = sums / l_size
    means = means * (1 - w)
    # binary = np.zeros(means.shape)
    # binary[means >= thres] = 1
    return img > means  # binary.astype(bool)


def _padding(img, padding, mode='constant'):
    return np.pad(img, padding, mode)


def estimate_Smap(img, binary):
    H, W = img.shape
    out = np.zeros(img.shape)
    for h in range(H):
        for w in range(W):
            if binary[h, w]:
                out[h, w] = img[h, w]
            else:
                if h > 0 and w > 0:
                    a = 1
                i = j = 2
                b = binary[(h - i) if (h - i) >= 0 else 0: h + i +
                           1, w - j if (w - j) >= 0 else 0: w + j + 1]
                counter = 0
                while np.sum(b) < 25:
                    # if counter % 2 == 0:
                    #     i += 1
                    # else:
                    #     j += 1
                    # counter += 1
                    i += 1
                    j += 1
                    b = binary[(h - i) if (h - i) >= 0 else 0: h +
                               i + 1, w - j if (w - j) >= 0 else 0: w + j + 1]
                window = img[(h - i) if (h - i) >= 0 else 0: h +
                             i + 1, w - j if (w - j) >= 0 else 0: w + j + 1]
                out[h, w] = (np.sum(window * b) / np.sum(b))
    return out


def shadow_removal_ICASSP2018(img, n_iter=1):
    # 1 for B, 0 for F
    HEIGHT = 2100
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        pass
    scale = HEIGHT/img.shape[0]
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    save_all_img = False
    img = img.astype(np.float32) / 255
    I = img.copy()
    binary = None
    for i in range(n_iter):
        binary = bradley_roth_threshold(I * 255)
        S = estimate_Smap(I, binary)
        R = I / S
        if np.linalg.norm(R - I) < 0.5:
            break
        I = np.clip(R, 0, 1)

        print("iter: %d/%d" % (i, n_iter))
        if save_all_img:
            cv2.imwrite('./B_%d.png' % i, (binary * 255).astype(np.uint8))
            cv2.imwrite('./out_%d.png' % i, (R * 255).astype(np.uint8))

    # if not save_all_img:
    #     cv2.imwrite('./B.png', (binary * 255).astype(np.uint8))
    #     cv2.imwrite('./out.png', (I * 255).astype(np.uint8))
    img = (I * 255).astype(np.uint8)
    if scale < 1:
        img = cv2.resize(img, (0, 0), fx=1/scale, fy=1/scale)
    return img


def main():
    img = cv2.imread('test.jpg')
    shadow_removal_ICASSP2018(img)


if __name__ == '__main__':
    main()
