import os
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def draw_hist(img):
    plt_y = np.zeros(256, dtype=int)
    for y in range(img.height):
        for x in range(img.width):
            plt_y[int(img.getpixel((x, y)))] += 1
    plt_x = np.arange(len(plt_y))
    plt.bar(plt_x, plt_y, width=1)
    plt.show()

def np_array_to_2d_grayscale_image(np_array, offset=0, detect_boundary=False):
    if(offset != 0):
        for y in range(np_array.shape[0]):
            for x in range(np_array.shape[1]):
                np_array[y, x] = np_array[y, x]+offset
    if(detect_boundary):
        for y in range(np_array.shape[0]):
            for x in range(np_array.shape[1]):
                if(np_array[y, x] < 0):
                    np_array[y, x] = 0
                elif(np_array[y, x] > 255):
                    np_array[y, x] = 255
    img = Image.fromarray(np_array.astype(np.uint8))
    img.save("ReferenceImage/sample3_2x2_binning_90_degrees.png")
    draw_hist(img)
    return img

def get_Sobel_edge_image(img, threshold):
    image_gradient = Image.new('L', (img.width, img.height))
    edge_map = Image.new('1', (img.width, img.height))
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((1, 1), (1, 1)), 'edge')
    k = 2
    for y in range(1, array_img.shape[0]-1):
        for x in range(1, array_img.shape[1]-1):
            gr = (1/(k+2))*((array_img[y-1,x+1]+k*array_img[y,x+1]+array_img[y+1,x+1])
                            -(array_img[y-1,x-1]+k*array_img[y,x-1]+array_img[y+1,x-1]))
            gc = (1/(k+2))*((array_img[y-1,x-1]+k*array_img[y-1,x]+array_img[y-1,x+1])
                            -(array_img[y+1,x-1]+k*array_img[y+1,x]+array_img[y+1,x+1]))
            g = (gr**2+gc**2)**0.5
            image_gradient.putpixel((x-1, y-1), int(g))
            if(g < threshold):
                edge_map.putpixel((x-1, y-1), 0)
            else:
                edge_map.putpixel((x-1, y-1), 1)
    return image_gradient,  edge_map

def get_Canny_edge_image(img, th, tl):
    edge_map = Image.new('1', (img.width, img.height))

    # step 1
    gaussian_filter_5x5_sigma_1_4 = np.array([[2 , 4 , 5 , 4 , 2 ],
                                              [4 , 9 , 12, 9 , 4 ],
                                              [5 , 12, 15, 12, 5 ],
                                              [4 , 9 , 12, 9 , 4 ],
                                              [2 , 4 , 5 , 4 , 2 ]], dtype=int)
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((2, 2), (2, 2)), 'edge')
    img_smooth = Image.new('L', (img.width, img.height))
    for y in range(2, array_img.shape[0]-2):
        for x in range(2, array_img.shape[1]-2):
            sum = 0
            for j in range(0, 5):
                for i in range(0, 5):
                    sum += array_img[y+j-2, x+i-2]*gaussian_filter_5x5_sigma_1_4[j, i]
            sum /= 159
            img_smooth.putpixel((x-2, y-2), int(sum))

    # step 2
    array_img_smooth = np.array(img_smooth, dtype=int)
    array_img_smooth_g = np.zeros((img_smooth.height, img_smooth.width), dtype=float)
    array_img_smooth_t = np.zeros((img_smooth.height, img_smooth.width), dtype=float)
    array_img_smooth = np.pad(array_img_smooth, ((1, 1), (1, 1)), 'edge')
    k = 2
    for y in range(1, array_img_smooth.shape[0]-1):
        for x in range(1, array_img_smooth.shape[1]-1):
            gr = (1/(k+2))*((array_img_smooth[y-1,x+1]+k*array_img_smooth[y,x+1]+array_img_smooth[y+1,x+1])
                            -(array_img_smooth[y-1,x-1]+k*array_img_smooth[y,x-1]+array_img_smooth[y+1,x-1]))
            gc = (1/(k+2))*((array_img_smooth[y-1,x-1]+k*array_img_smooth[y-1,x]+array_img_smooth[y-1,x+1])
                            -(array_img_smooth[y+1,x-1]+k*array_img_smooth[y+1,x]+array_img_smooth[y+1,x+1]))
            array_img_smooth_g[y-1,x-1] = (gr**2+gc**2)**0.5
            if(gr == 0):                                    # avoid divide by zero
                gr = 10e-8
            array_img_smooth_t[y-1,x-1] = math.atan(gc/(gr))

    # step 3
    array_img_smooth_g_new = np.copy(array_img_smooth_g)
    array_img_smooth_g = np.pad(array_img_smooth_g, ((1, 1), (1, 1)), 'edge')
    for y in range(1, array_img_smooth_g.shape[0]-1):
        for x in range(1, array_img_smooth_g.shape[1]-1):
            degree = int(math.degrees(array_img_smooth_t[y-1,x-1])%180)
            if(degree < 0):
                degree += 180
            if(degree < 22.5 or degree > 157.5):
                if(array_img_smooth_g[y, x] <= array_img_smooth_g[y, x+1]
                   or array_img_smooth_g[y, x] <= array_img_smooth_g[y, x-1]):
                    array_img_smooth_g_new[y-1, x-1] = 0
            elif(degree >= 22.5 and degree < 67.5):
                if(array_img_smooth_g[y, x] <= array_img_smooth_g[y-1, x+1]
                   or array_img_smooth_g[y, x] <= array_img_smooth_g[y+1, x-1]):
                    array_img_smooth_g_new[y-1, x-1] = 0
            elif(degree >= 67.5 and degree < 112.5):
                if(array_img_smooth_g[y, x] <= array_img_smooth_g[y-1, x]
                   or array_img_smooth_g[y, x] <= array_img_smooth_g[y+1, x]):
                    array_img_smooth_g_new[y-1, x-1] = 0
            elif(degree >= 112.5 and degree < 157.5):
                if(array_img_smooth_g[y, x] <= array_img_smooth_g[y-1, x-1]
                   or array_img_smooth_g[y, x] <= array_img_smooth_g[y+1, x+1]):
                    array_img_smooth_g_new[y-1, x-1] = 0
    
    # step 4
    array_img_edge = np.zeros((img.height, img.width), dtype=int)
    for y in range(0, array_img_smooth_g_new.shape[0]):
        for x in range(0, array_img_smooth_g_new.shape[1]):
            if(array_img_smooth_g[y, x] >= th):     # Edge Pixel
                array_img_edge[y, x] = 2
            elif(array_img_smooth_g[y, x] < tl):    # Non-edge Pixel
                array_img_edge[y, x] = 0
            else:                                   # Candidate Pixel
                array_img_edge[y, x] = 1
    
    # step 5
    array_img_edge = np.pad(array_img_edge, ((1, 1), (1, 1)), 'edge')
    for y in range(1, array_img_edge.shape[0]-1):
        for x in range(1, array_img_edge.shape[1]-1):
            if(array_img_edge[y, x] == 2):
                edge_map.putpixel((x-1, y-1), 1)
            elif(array_img_edge[y, x] == 0):
                edge_map.putpixel((x-1, y-1), 0)
            else:
                if(array_img_edge[y-1, x-1] > 0 or array_img_edge[y-1, x] > 0 or
                   array_img_edge[y-1, x+1] > 0 or array_img_edge[y, x-1] > 0 or
                   array_img_edge[y, x+1] > 0 or array_img_edge[y+1, x-1] > 0 or
                   array_img_edge[y+1, x] > 0 or array_img_edge[y+1, x+1] > 0):
                    edge_map.putpixel((x-1, y-1), 1)
                else:
                    edge_map.putpixel((x-1, y-1), 0)
    return edge_map

def get_LoG_edge_image(img, threshold):
    edge_map = Image.new('1', (img.width, img.height))

    # step 1
    gaussian_filter_5x5 = np.array([[1 , 4 , 7 , 4 , 1 ],
                                    [4 , 16, 26, 16, 4 ],
                                    [7 , 26, 41, 26, 7 ],
                                    [4 , 16, 26, 16, 4 ],
                                    [1 , 4 , 7 , 4 , 1 ]], dtype=int)
    array_img = np.array(img, dtype=int)
    array_img = np.pad(array_img, ((2, 2), (2, 2)), 'edge')
    img_smooth = Image.new('L', (img.width, img.height))
    for y in range(2, array_img.shape[0]-2):
        for x in range(2, array_img.shape[1]-2):
            sum = 0
            for j in range(0, 5):
                for i in range(0, 5):
                    sum += array_img[y+j-2, x+i-2]*gaussian_filter_5x5[j, i]
            sum /= 273
            img_smooth.putpixel((x-2, y-2), int(sum))
    
    # step 2
    high_pass_filter_3x3 = np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]], dtype=int)
    array_img_smooth = np.array(img_smooth, dtype=int)
    array_img_smooth = np.pad(array_img_smooth, ((1, 1), (1, 1)), 'edge')
    array_img_LoG = np.zeros((img_smooth.height, img_smooth.width), dtype=int)
    for y in range(1, array_img_smooth.shape[0]-1):
        for x in range(1, array_img_smooth.shape[1]-1):
            sum = 0
            for j in range(0, 3):
                for i in range(0, 3):
                    sum += array_img_smooth[y+j-1, x+i-1]*high_pass_filter_3x3[j, i]
            array_img_LoG[y-1, x-1] = sum/8
    
    # step 3: set up a threshold to separate zero and non-zero to get  G'
    for y in range(array_img_LoG.shape[0]):
        for x in range(array_img_LoG.shape[1]):
            if(array_img_LoG[y, x] <= threshold and array_img_LoG[y, x] >= -threshold):
                array_img_LoG[y, x] = 0
    
    # step 4: decide whether (j,k) is a zero-crossing point
    array_img_LoG = np.pad(array_img_LoG, ((1, 1), (1, 1)), 'edge')
    for y in range(1, array_img_LoG.shape[0]-1):
        for x in range(1, array_img_LoG.shape[1]-1):
            if(array_img_LoG[y, x] == 0):
                if(array_img_LoG[y, x-1]*array_img_LoG[y, x+1] < 0 or 
                   array_img_LoG[y+1, x-1]*array_img_LoG[y-1, x+1] < 0 or 
                   array_img_LoG[y-1, x]*array_img_LoG[y+1, x] < 0 or 
                   array_img_LoG[y-1, x-1]*array_img_LoG[y+1, x+1] < 0):
                    edge_map.putpixel((x-1, y-1), 1)
                else:
                    edge_map.putpixel((x-1, y-1), 0)
            else:
                edge_map.putpixel((x-1, y-1), 0)
    return edge_map

def get_edge_crispening_image(img, c):      # 0.6 <= c <= 0.83333
    img_g = Image.new('L', (img.width, img.height))
    low_pass_filter_3x3 = np.array([[1, 1, 1],
                                    [1, 2, 1],
                                    [1, 1, 1]], dtype=int)
    array_img = np.array(img, dtype=int)
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
    return img_g

def get_morph_optim_image(img):                         # 8-neighbor dilate
    img_optim_1 = img.copy()
    for y in range(1, img.height-1):
        for x in range(1, img.width-1):
            if(img.getpixel((x, y)) > 0):
                if(img.getpixel((x-1, y-1)) == 0 or img.getpixel((x, y-1)) == 0 or img.getpixel((x+1, y-1)) == 0 or
                   img.getpixel((x-1, y)) == 0 or img.getpixel((x+1, y)) == 0 or img.getpixel((x-1, y+1)) == 0 or
                   img.getpixel((x, y+1)) == 0 or img.getpixel((x+1, y+1)) == 0):
                    img_optim_1.putpixel((x, y), 0)
    img_optim_2 = img_optim_1.copy()                    # remove isolated pixel
    for y in range(1, img_optim_1.height-1):
        for x in range(1, img_optim_1.width-1):
            if(img_optim_1.getpixel((x, y)) > 0):
                count = 0
                for j in range(0, 3):
                    for i in range(0, 3):
                        if(img_optim_1.getpixel((x+i-1, y+j-1)) == 0):
                            count += 1
                if(count >= 5):
                    img_optim_2.putpixel((x, y), 0)
    return img_optim_2

def get_4_cat_image(img):
    img_4_cat = Image.new('L', (img.width, img.height))

    # shrink 1/2 - up cat
    array_cat_1 = np.zeros((img.height//2, img.width//2), dtype=int)
    for y in range(0, img.height, 2):
        for x in range(0, img.width, 2):
            mean = (img.getpixel((x, y))+img.getpixel((x+1, y))+img.getpixel((x, y+1))+img.getpixel((x+1, y+1)))//4
            array_cat_1[y//2,x//2] = mean
    for y in range(array_cat_1.shape[0]):
        for x in range(array_cat_1.shape[1]):
            if(array_cat_1[y, x] > 0):
                img_4_cat.putpixel((x+143, y+25), int(array_cat_1[y, x]))

    # rotate 180 degrees - bottom cat
    array_cat_2 = np.zeros((img.height//2, img.width//2), dtype=int)
    for y in range(array_cat_1.shape[0]):
        for x in range(array_cat_1.shape[1]):
            x_rotate = -x
            y_rotate = -y
            array_cat_2[y_rotate+299, x_rotate+299] = array_cat_1[y, x]
    for y in range(array_cat_2.shape[0]):
        for x in range(array_cat_2.shape[1]):
            if(array_cat_2[y, x] > 0):
                img_4_cat.putpixel((x+157, y+275), int(array_cat_2[y, x]))

    # rotate -90 degrees - left cat
    array_cat_3 = np.zeros((img.height//2, img.width//2), dtype=int)
    for y in range(array_cat_1.shape[0]):
        for x in range(array_cat_1.shape[1]):
            x_rotate = y
            y_rotate = -x
            array_cat_3[y_rotate+299, x_rotate] = array_cat_1[y, x]
    for y in range(array_cat_3.shape[0]):
        for x in range(array_cat_3.shape[1]):
            if(array_cat_3[y, x] > 0):
                img_4_cat.putpixel((x+25, y+157), int(array_cat_3[y, x]))
    
    # rotate 90 degrees - right cat
    array_cat_4 = np.zeros((img.height//2, img.width//2), dtype=int)
    for y in range(array_cat_1.shape[0]):
        for x in range(array_cat_1.shape[1]):
            x_rotate = -y
            y_rotate = x
            array_cat_4[y_rotate, x_rotate+299] = array_cat_1[y, x]
    for y in range(array_cat_4.shape[0]):
        for x in range(array_cat_4.shape[1]):
            if(array_cat_4[y, x] > 0):
                img_4_cat.putpixel((x+275, y+143), int(array_cat_4[y, x]))
    return img_4_cat

def get_warp_cat_image(img):
    img_warp_x = Image.new('L', (img.width, img.height))
    img_warp_cat = Image.new('L', (img.width, img.height))
    for y in range(img.height):
        for x in range(img.width):
            if(img.getpixel((x, y)) > 0):
                img_warp_x.putpixel((x, int(23*math.sin(x/23-5)+y)), img.getpixel((x, y)))
    for y in range(img.height):
        for x in range(img.width):
            if(img_warp_x.getpixel((x, y)) > 0):
                img_warp_cat.putpixel((int(23*math.sin(y/23-10)+x), y), img_warp_x.getpixel((x, y)))
    return img_warp_cat

def get_huge_space(edge_map):
    plt.title('Huge Space')
    plt.xlabel('theta')
    plt.ylabel('rho')
    theta = np.linspace(-np.pi, np.pi, num=50)
    for y in range(edge_map.height):
        for x in range(edge_map.width):
            if(edge_map.getpixel((x, y)) > 0):
                a1 = math.sin(theta)
                a2 = x*a1
                a3 = y*a2
                rho = a2+a3
                # rho = x*math.sin(theta)+y*math.sin(theta)
                plt.plot(theta,rho)
    plt.show()
    plt.savefig("ResultImage/result6.png")

if __name__=='__main__':
    # read image & create result folder
    sample1 = Image.open("SampleImage/sample1.png")
    sample2 = Image.open("SampleImage/sample2.png")
    sample3 = Image.open("SampleImage/sample3.png")
    sample5 = Image.open("SampleImage/sample5.png").convert('L')
    os.makedirs('ResultImage', exist_ok=True)
    
    result1, result2 = get_Sobel_edge_image(sample1, threshold=40)
    result1.save("ResultImage/result1.png")
    result2.save("ResultImage/result2.png")

    result3 = get_Canny_edge_image(sample1, th=50, tl=25)
    result3.save("ResultImage/result3.png")
    '''
    result4 = get_LoG_edge_image(sample1, threshold=5)
    result4.save("ResultImage/result4.png")

    result5 = get_edge_crispening_image(sample2, c=0.65)
    result5.save("ResultImage/result5.png")
    '''
    get_huge_space(result3)

    # sample3_optim = get_morph_optim_image(sample3)
    # sample3_optim.save("ReferenceImage/sample3_optim.png")
    '''
    result7 = get_4_cat_image(sample3)
    result7.save("ResultImage/result7.png")
    
    result8 = get_warp_cat_image(sample5)
    result8.save("ResultImage/result8.png")
    '''
