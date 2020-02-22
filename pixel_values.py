import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def image_pixels(image_path):
    image = Image.open(image_path, 'r')     # .open is a method in Image Class (Doesn't Give any data though).
    width, height = image.size
    pixel_values = list(image.getdata())    # returns the RGB values for a grid (UnRolled Form).
    pixel_values = np.array(pixel_values).reshape([width, height, 3])   # Rolled Form based on size of image.
    return pixel_values

im_pixels = image_pixels('/Users/nikhil/Desktop/Project/Image-Classification/Utilities/Car.jpg')
plt.figure()
plt.imshow(im_pixels)
plt.colorbar()
plt.show()