import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import math

# open sample 1 image
sample1 = Image.open("sample1.png")

# flip, Img mode 'L' for grayscale / "RGB" for RGB
result1 = Image.new('RGB', (sample1.width, sample1.height))
for y in range(result1.height):
    for x in range(result1.width):
        color = sample1.getpixel((x, y))
        result1.putpixel((result1.width-1-x, y), color)
result1.save("result1.png")

# by definition: "Gray = 0.299 * Red + 0.587 * Green + 0.114 * Blue"
result2 = Image.new('L', (sample1.width, sample1.height))
for y in range(result2.height):
    for x in range(result2.width):
        r, g, b = result1.getpixel((x, y))
        gray = 0.299*r+0.587*g+0.114*b
        result2.putpixel((x, y), int(gray))
result2.save("result2.png")

# open sample 2 image
sample2 = Image.open("sample2.png")

# intensity/2 
result3 = Image.new('L', (sample2.width, sample2.height))
for y in range(result3.height):
    for x in range(result3.width):
        gray = sample2.getpixel((x, y))
        result3.putpixel((x, y), int(gray/2))
result3.save("result3.png")

# intensity*3
result4 = Image.new('L', (sample2.width, sample2.height))
for y in range(result4.height):
    for x in range(result4.width):
        gray = result3.getpixel((x, y))
        gray = gray*3
        if(gray > 255):
            gray = 255
        result4.putpixel((x, y), int(gray))
result4.save("result4.png")

# plot function
def draw_histogram(img):
    plt_y = np.zeros(256, dtype=int)
    for y in range(img.height):
        for x in range(img.width):
            plt_y[int(img.getpixel((x, y)))] += 1
    plt_x = np.arange(len(plt_y))
    plt.bar(plt_x, plt_y, width=1)
    plt.show()

# plot sample2, result3 and result4
draw_histogram(sample2)
draw_histogram(result3)
draw_histogram(result4)

# global histogram equalization function
def global_histogram_equalization(img):
    img_new = Image.new('L', (img.width, img.height))
    hist = np.zeros(256, dtype=int)
    for y in range(img.height):
        for x in range(img.width):
            hist[int(img.getpixel((x, y)))] += 1
    pdf = np.zeros(256, dtype=float)
    for i in range(len(pdf)):
        pdf[i] = hist[i]/(img.width*img.height)
    cdf = np.zeros(256, dtype=float)
    cdf[0] = pdf[0]*255
    for i in range(1, len(cdf)):
        cdf[i] = cdf[i-1]+pdf[i]*255
    for y in range(img_new.height):
        for x in range(img_new.width):
            gray = round(cdf[int(img.getpixel((x, y)))])
            img_new.putpixel((x, y), gray)
    return img_new

# global histogram equalization & plot
result5 = global_histogram_equalization(result3)
result5.save("result5.png")
result6 = global_histogram_equalization(result4)
result6.save("result6.png")
draw_histogram(result5)
draw_histogram(result6)

# local histogram equalization function
# ref: https://www.imageeprocessing.com/2011/06/local-histogram-equalization.html
def local_histogram_equalization(img):
    img_new = Image.new('L', (img.width, img.height))
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((1, 1), (1, 1)), 'edge')
    for y in range(1, array_img.shape[0]-1):
        for x in range(1, array_img.shape[1]-1):
            mask = np.zeros((3, 3), dtype=int)
            for j in range(y-1, y+2):
                for i in range(x-1, x+2):
                    mask[j-(y-1), i-(x-1)] = array_img[j, i]
            hist = np.zeros(256, dtype=int)
            for j in range(mask.shape[0]):
                for i in range(mask.shape[1]):
                    hist[mask[j, i]] += 1
            pdf = np.zeros(256, dtype=float)
            for i in range(len(pdf)):
                pdf[i] = hist[i]/(mask.shape[1]*mask.shape[0])
            cdf = np.zeros(256, dtype=float)
            cdf[0] = pdf[0]*255
            for i in range(1, len(cdf)):
                cdf[i] = cdf[i-1]+pdf[i]*255
            gray = round(cdf[array_img[y, x]])
            img_new.putpixel((x-1, y-1), gray)
    return img_new

# local histogram equalization & plot
result7 = local_histogram_equalization(result3)
result7.save("result7.png")
result8 = local_histogram_equalization(result4)
result8.save("result8.png")
draw_histogram(result7)
draw_histogram(result8)

# optimize, goal: improve low intensity part
result9 = Image.new('L', (sample2.width, sample2.height))
for y in range(result9.height):
    for x in range(result9.width):
        gray = int(sample2.getpixel((x, y)))
        if(gray < 147):
            gray = gray**(2/5)*20
        result9.putpixel((x, y), int(gray))
result9.save("result9.png")
draw_histogram(result9)

# open sample 3,4,5 image
sample3 = Image.open("sample3.png")
sample4 = Image.open("sample4.png")
sample5 = Image.open("sample5.png")

#
def noise_reduction(img, filter_size):
    img_new = Image.new('L', (img.width, img.height))
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((filter_size[0]//2, filter_size[0]//2), (filter_size[1]//2, filter_size[1]//2)), 'edge')
    for y in range(filter_size[0]//2, array_img.shape[0]-filter_size[0]//2):
        for x in range(filter_size[1]//2, array_img.shape[1]-filter_size[1]//2):
            array_filter = np.zeros((filter_size[0], filter_size[1]), dtype=int)
            for j in range(y-filter_size[0]//2, y+filter_size[0]//2+1):
                for i in range(x-filter_size[1]//2, x+filter_size[1]//2+1):
                    array_filter[j-(y-filter_size[0]//2), i-(x-filter_size[1]//2)] = array_img[j, i]
            gray = int(np.median(array_filter))
            if(gray == 255):
                gray = int(np.percentile(array_filter, 25))
            elif(gray == 0):
                gray = int(np.percentile(array_filter, 75))
            img_new.putpixel((x-filter_size[1]//2, y-filter_size[0]//2), gray)
    return img_new

result10 = noise_reduction(sample4, np.array([3, 3]))
result10.save("result10.png")
result11 = noise_reduction(sample5, np.array([3, 3]))
result11.save("result11.png")

def psnr(base_img, processed_img):
    mse = 0
    for y in range(base_img.height):
        for x in range(base_img.width):
            mse += (int(base_img.getpixel((x, y)))-int(processed_img.getpixel((x, y))))**2
    mse /= base_img.width*base_img.height
    return 10*math.log(255**2/mse, 10)

print("PSNR of result10 is: %f" %(psnr(sample3, result10)))
print("PSNR of result11 is: %f" %(psnr(sample3, result11)))
