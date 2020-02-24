import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import os

def image_pixels(img_path):                 # Returns A List Of The Pixel Values For A Given Image
    image = Image.open(img_path, 'r')       # .open is a method in Image Class (Doesn't Give any data though).
    width, height = image.size
    print image.size
    pixel_values = list(image.getdata())    # returns the RGB values for a grid (UnRolled Form).
    pixel_values = np.array(pixel_values).reshape([width, height, 3])   # Rolled Form based on size of image.
    return pixel_values

def get_images(directory):                  # Returns A Numpy Array Of The Path Of Images
    image_files = []
    directory = os.path.join(directory, 'Train')
    for img_file in os.listdir(directory):
        if img_file.endswith('.jpg'):
            image_files.append(os.path.join(directory, img_file))
    image_files = np.array(sorted(image_files))
    return image_files

def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display_3(a,b,c):
    plt.subplot(131), plt.imshow(a), plt.title('HSV')
    plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(b), plt.title('Mask')
    plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(c), plt.title('Img')
    plt.xticks([]), plt.yticks([])
    plt.show()

def disp_transf(a,b,c,d):
    plt.subplot(141), plt.imshow(a), plt.title('Resized Image')
    plt.xticks([]), plt.yticks([])
    plt.subplot(142), plt.imshow(b), plt.title('Segmented')
    plt.xticks([]), plt.yticks([])
    plt.subplot(143), plt.imshow(c), plt.title('Opening')
    plt.xticks([]), plt.yticks([])
    plt.subplot(144), plt.imshow(d), plt.title('Dilate')
    plt.xticks([]), plt.yticks([])
    plt.show()

def image_processing(img_path):
    # Resize Image: (Baseline Size For All Images Fed Into Our AI Algorithm)
    # ------------------------------------------------------------------------------------------------------------------
    img_value = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    print ('Original Value = ', img_value.shape)    # (Height, Width, RGB)
    width  = 256    # Resized Values
    height = 256    # Resized Values
    dim = (width, height)
    res_img = [cv2.resize(img_value, dim, interpolation=cv2.INTER_CUBIC)]
    print ('ReSized: ', res_img[0].shape)
    #display(img_value, res_img[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Removing Noise Using GaussianBlur: (To Remove Noise In Our Data, Obtain Uniform Variation In Brightness/Color)
    image_no_noise = []
    gaussian_blur = cv2.GaussianBlur(res_img[0], (5, 5), 0)
    image_no_noise.append(gaussian_blur)
    #display(res_img[0], image_no_noise[0], 'ReSized', 'Blurred')

    # ------------------------------------------------------------------------------------------------------------------
    # Segmentation (GrayScaled -> AdaptiveGaussianThreshold)

    ''' Color Really Doesnt Help Us In Image Classification Of Vehicles, A More Important Prospect For This Problem Is
     How The Edges And Shape Change, Hence Only The Information On " HOW THE BOUNDARIES ARE CHANGING IS WHAT MATTERS
     IN THIS PARTICULAR PROBLEM STATEMENT ", had it been color related then it would have been a different scenario.'''

    hsv_image = cv2.cvtColor(image_no_noise[0], cv2.COLOR_BGR2HSV)  # Converts The Image Into 'Gray Scaled.
    lower_red = np.array([70, 15, 0])
    upper_red = np.array([255,255,255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)  # Would Mask If A Particular Grids Pixel Would Fall In The Limit
    res = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    display_one(mask)

    # Separating Objects With Markers:

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(res_img[0], markers)
    res_img[0][markers == -1] = [255, 0, 0]
    disp_transf(res_img[0], hsv_image, mask, markers)

    #display(res_img[0], markers, 'Original', 'Marked')

path = '/Users/nikhil/Desktop/Project/Image-Classification/Utilities'
image_path = get_images(path)
image_processing(image_path[120])
