# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
from sklearn.cluster import KMeans
import numpy as np
from matplotlib.figure import Figure
import cv2
import io

# ---------------------------------------------------------------------------- #
#                                Hyperparameters                               #
# ---------------------------------------------------------------------------- #
RESIZE_HEIGHT = 500
FILTER_SIZE = 11
FILTER_SIGMA_COLOR = 75
FILTER_SIGMA_SPACE = 75
ADAPTIVE_BLOCKSIZE = 11
ADAPTIVE_C = 2
HOUGH_LENGTH = 0.3 * RESIZE_HEIGHT
A4_POINTS = np.array([(0, 0), (2100, 0), (2100, 2970), (0, 2970)], np.float32)
A5_POINTS = np.array([(0, 0), (1480, 0), (1480, 2100), (0, 2100)], np.float32)
B4_POINTS = np.array(
    [(0, 0), (2500*0.85, 0), (2500*0.85, 3530*0.85), (0, 3500*0.85)], np.float32)
B5_POINTS = np.array([(0, 0), (1760, 0), (1760, 2500), (0, 2500)], np.float32)
S9_16_POINTS = np.array(
    [(0, 0), (2160*0.8, 0), (2160*0.8, 3840*0.8), (0, 3840*0.8)], np.float32)
S3_4_POINTS = np.array(
    [(0, 0), (3072*0.7, 0), (3072*0.7, 4096*0.7), (0, 4096*0.7)], np.float32)

# ---------------------------------------------------------------------------- #
#                               Utility Functions                              #
# ---------------------------------------------------------------------------- #


def intersection_rho_theta(rho1, theta1, rho2, theta2):
    """
    Find the intersection of two lines using rho-theta representation.
    Ref: https://stackoverflow.com/questions/383480/intersection-of-two-lines-defined-in-rho-theta-parameterization
    """
    ct1 = np.cos(theta1)
    st1 = np.sin(theta1)
    ct2 = np.cos(theta2)
    st2 = np.sin(theta2)
    denom = ct1 * st2 - ct2 * st1
    if denom == 0:
        return None
    else:
        x = (st2 * rho1 - st1 * rho2) / denom
        y = (ct1 * rho2 - ct2 * rho1) / denom
        return x, y


def order_points(pts):
    """
    Order the points in a clockwise fashion.
    Ref: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def draw_hough_lines(img, lines, color=(0, 0, 255), thickness=1, groups=None):
    if groups is not None:
        group_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                        (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        if groups is not None:
            cv2.line(img_colored, pt1, pt2,
                     group_colors[groups[i]], thickness, cv2.LINE_AA)
        else:
            cv2.line(img_colored, pt1, pt2, color, thickness, cv2.LINE_AA)
    return img_colored


def draw_points(img, points, color=(0, 0, 255), thickness=5):
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(points)):
        pt = points[i]
        cv2.circle(img_colored, pt, thickness, color, -1)
    return img_colored


def convert_figure_to_array(figure):
    """
    Convert a Matplotlib figure to a numpy array
    Ref: https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
    """
    with io.BytesIO() as buff:
        figure.savefig(buff, format='raw')
        buff.seek(0)
        data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    im = data.reshape((int(h), int(w), -1))
    return im


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
# ---------------------------------------------------------------------------- #
#                                  Main Logic                                  #
# ---------------------------------------------------------------------------- #


def crop_and_transform(img, pageSize='A4'):
    """
    Crop the image and transform it to a Rectangle.
    """
    # cv2.imwrite('Original.jpg', img)
    # ----------------- Setup Figure to Record Intermediate Steps ---------------- #
    fig = Figure(figsize=(15, 8))

    # ------------------------------- Preprocessing ------------------------------ #
    scale = RESIZE_HEIGHT / img.shape[0]
    resized = cv2.resize(img, (int(img.shape[1] * scale), RESIZE_HEIGHT))
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ax = fig.add_subplot(2, 4, 1)
    ax.imshow(grayscale, cmap='gray')
    ax.set_title('Grayscale')

    # --------------------------------- Binarize --------------------------------- #
    blurred = cv2.bilateralFilter(
        grayscale, FILTER_SIZE, FILTER_SIGMA_COLOR, FILTER_SIGMA_SPACE)
    equalized = cv2.createCLAHE(
        clipLimit=2.0, tileGridSize=(8, 8)).apply(blurred)
    ax = fig.add_subplot(2, 4, 2)
    ax.imshow(equalized, cmap='gray')
    ax.set_title('Equilized')
    # cv2.imwrite('Equalized.jpg', equalized)

    thresh = cv2.adaptiveThreshold(
        equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, ADAPTIVE_BLOCKSIZE, ADAPTIVE_C)
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    ax = fig.add_subplot(2, 4, 3)
    ax.imshow(thresh, cmap='gray')
    ax.set_title('Adaptive Threshold')
    # cv2.imwrite('Threshold.jpg', thresh)

    # ------------------- Find the Largest Connected Component ------------------- #

    components = cv2.connectedComponentsWithStats(thresh)
    max_area = 0
    max_idx = 0
    for i in range(1, len(components[2])):
        if components[2][i][4] > max_area:
            max_area = components[2][i][4]
            max_idx = i
    filtered = np.zeros(thresh.shape, np.uint8)
    filtered[components[1] == max_idx] = 255
    filtered = cv2.morphologyEx(
        filtered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    ax = fig.add_subplot(2, 4, 4)
    ax.imshow(filtered, cmap='gray')
    ax.set_title('Largest Connected Component')
    # cv2.imwrite('Component.jpg', filtered)

    # --------------------- Find the Border of the Component --------------------- #
    contour, hier = cv2.findContours(
        filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_idx = 0
    for i in range(len(contour)):
        area = cv2.contourArea(contour[i])
        if area > max_area:
            max_area = area
            max_idx = i
    contour = contour[max_idx]
    contour_image = np.zeros(grayscale.shape, np.uint8)
    cv2.drawContours(contour_image, [contour], 0, 255, 2)
    ax = fig.add_subplot(2, 4, 5)
    ax.imshow(contour_image, cmap='gray')
    ax.set_title('Contour')
    # cv2.imwrite('Contour.jpg', contour_image)

    # ------------------------------ Hough Transform ----------------------------- #
    lines = cv2.HoughLines(
        contour_image, 1, np.pi / 180, round(HOUGH_LENGTH), None, 0, 0)
    lines_new = []
    # drop lines at four edges
    for line in lines:
        rho, theta = line[0]
        if ((theta == 0 or theta == np.pi) and (rho == 0 or rho == grayscale.shape[1]-1)) or (theta == np.pi/2 and (rho == 0 or rho == grayscale.shape[0]-1)):
            continue
        lines_new.append(line)
    lines = lines_new
    ax = fig.add_subplot(2, 4, 6)
    ax.imshow(draw_hough_lines(grayscale, lines))
    ax.set_title('Hough Transform')
    # cv2.imwrite('Hough.jpg', draw_hough_lines(grayscale, lines))

    # ------------------ Cluster the Lines Use the Center Points ----------------- #
    center = (int(grayscale.shape[1] / 2), int(grayscale.shape[0] / 2))
    points = []
    for i in range(0, len(lines)):
        # get the point on the line that is closest to the center
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        s = (-b*center[0] + a*center[1] + b * x0 - a * y0) / (a ** 2 + b ** 2)
        x = int(x0 + s * -b)
        y = int(y0 + s * a)
        points.append((x, y))

    kmeans = KMeans(n_clusters=4, random_state=0).fit_predict(points)
    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(draw_hough_lines(grayscale, lines, groups=kmeans))
    ax.set_title('Clustered Lines')

    # cv2.imwrite('Clustered_line.jpg', draw_hough_lines(
    #     grayscale, lines, groups=kmeans))
    # cv2.imwrite('Clustered_center.jpg', draw_points(grayscale, points))

    # -------- Cluster the Intersection Points of the Lines into 4 Groups -------- #
    intersections = []
    for i in range(0, len(kmeans)):
        for j in range(i+1, len(kmeans)):
            if kmeans[i] != kmeans[j]:
                point = intersection_rho_theta(
                    lines[i][0][0], lines[i][0][1], lines[j][0][0], lines[j][0][1])
                if point is not None:
                    x, y = point
                    if 0 <= x < grayscale.shape[1] and 0 <= y < grayscale.shape[0]:
                        intersections.append(point)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(
        intersections).cluster_centers_
    kmeans = order_points(np.array(kmeans, np.float32))
    ax = fig.add_subplot(2, 4, 8)
    ax.imshow(draw_points(grayscale, kmeans.astype(np.int32)))
    ax.set_title('Clustered Intersections')
    # cv2.imwrite('Clustered_intersections.jpg',
    #             draw_points(grayscale, kmeans.astype(np.int32)))

    # ----------------------- Perform Perspective Transfrom ---------------------- #
    if pageSize == 'A4':
        homography = cv2.getPerspectiveTransform(kmeans / scale, A4_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(A4_POINTS[2, 0]), round(A4_POINTS[2, 1])))
    elif pageSize == 'A5':
        homography = cv2.getPerspectiveTransform(kmeans / scale, A5_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(A5_POINTS[2, 0]), round(A5_POINTS[2, 1])))
    elif pageSize == 'B4':
        homography = cv2.getPerspectiveTransform(kmeans / scale, B4_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(B4_POINTS[2, 0]), round(B4_POINTS[2, 1])))
    elif pageSize == 'B5':
        homography = cv2.getPerspectiveTransform(kmeans / scale, B5_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(B5_POINTS[2, 0]), round(B5_POINTS[2, 1])))
    elif pageSize == '9_16':
        homography = cv2.getPerspectiveTransform(kmeans / scale, S9_16_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(S9_16_POINTS[2, 0]), round(S9_16_POINTS[2, 1])))
    elif pageSize == '3_4':
        homography = cv2.getPerspectiveTransform(kmeans / scale, S3_4_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(S3_4_POINTS[2, 0]), round(S3_4_POINTS[2, 1])))
    else:
        homography = cv2.getPerspectiveTransform(kmeans / scale, A4_POINTS)
        warped = cv2.warpPerspective(
            img, homography, (round(A4_POINTS[2, 0]), round(A4_POINTS[2, 1])))

    return convert_figure_to_array(fig), warped


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--image', type=str, required=True)
    args = argparser.parse_args()
    img = cv2.imread(args.image)
    output, result = crop_and_transform(img)
    cv2.imwrite('output.png', output)
    cv2.imwrite('result.png', result)
