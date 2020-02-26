# Image PreProcessing For Classifying Automobile's (4 Wheelers, 2 Wheelers, HCV, LCV, Others)

''' Credits To Prince Canuma, (Refrence: https://towardsdatascience.com/image-pre-processing-c1aec0be3edf)
For Detailing All The Necessary Steps Required In Image PreProcessing, 'THE SEGMENTATION METHOD USED IN 
THE ARTICLE DOESNT HELP FOR OUR PROBLEM STATEMENT', Hence A Different Approach Has Been Taken By Detecting
Edges Through Canny's Method, Appreciable Results Were Found '''

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

def grab_cut(img):
    rect = (5, 25, 230, 250)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask ==2) | (mask == 0), 0, 1).astype('uint8')
    img = img*mask2[:, :, np.newaxis]
    return img

def get_images(directory):                  # Returns A Numpy Array Of The Path Of Images
    image_files = []
    directory = os.path.join(directory, 'Train')
    for img_file in os.listdir(directory):
        if img_file.endswith('.jpg'):
            image_files.append(os.path.join(directory, img_file))
    image_files = np.array(sorted(image_files))
    return image_files

def save_images(directory, img_list):
    for i in range(len(img_list) - 1):
        cv2.imwrite(os.path.join(directory, str(i) + '.jpg'), img_list[i])
    print 'Successfully Saved.'

def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

def display(a, b, title1 = "Original", title2 = "Edited"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.xticks([]), plt.yticks([])
    plt.pause(2)
    plt.show(block=False)
    plt.close('all')


def image_processing(img_path):
    # Resize Image: (Baseline Size For All Images Fed Into Our ML Algorithm)
    # ------------------------------------------------------------------------------------------------------------------
    res_img = []
    img_value = []
    width  = 256    # Resized Values
    height = 256    # Resized Values
    dim = (width, height)
    for i in range(len(img_path) - 1):
        print i
        img_value.append(cv2.imread(img_path[i], cv2.IMREAD_UNCHANGED))
        res_img.append(cv2.resize(img_value[i], dim, interpolation=cv2.INTER_CUBIC))

    # ------------------------------------------------------------------------------------------------------------------
    # Removing Noise Using GaussianBlur: (To Remove Noise In Our Data, Obtain Uniform Variation In Brightness/Color)
    image_no_noise = []
    for i in range(len(img_path) - 1):
        gaussian_blur = cv2.GaussianBlur(res_img[i], (5, 5), 0)
        image_no_noise.append(gaussian_blur)
    #display(res_img[0], image_no_noise[0], 'ReSized', 'Blurred')

    # ------------------------------------------------------------------------------------------------------------------
    # Segmentation & Morphology:

    ''' Color Really Doesnt Help Us In Image Classification Of Vehicles, A More Important Prospect For This Problem Is
     How The Edges And Shape Change, Hence Only The Information On " HOW THE BOUNDARIES ARE CHANGING IS WHAT MATTERS
     IN THIS PARTICULAR PROBLEM STATEMENT ", had it been color related then it would have been a different scenario.'''

    modif_img = []
    for i in range(len(img_path) - 1):
        try:
            print i
            img = grab_cut(image_no_noise[i])  # Cuts The Resized Blurred Image Into The Rectangle Specified, Removes Noise To Some Extent
            edges = cv2.Canny(img, 100, 100)                      # Gets The Edges Of The Given Grab-Cut Image
            modif_img.append(cv2.GaussianBlur(edges, (5, 5), 0))  # Blurring The Image For Again Better Noise Removal
        except Exception:
            pass

    save_images(out_path, modif_img)

    #for i in range(len(img_path) - 1):
    #    display(res_img[i], modif_img[i], 'Resized_Img', 'Segmented Image')

in_path  = '/Users/nikhil/Desktop/Project/Image-Classification/Utilities'
out_path = '/Users/nikhil/Desktop/Project/Image-Classification/Utilities/Output_Files'
image_path = get_images(in_path)
image_processing(image_path[0:])
