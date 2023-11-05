import cv2
import numpy as np

# Load the image
image = cv2.imread('/Users/adityamaindan/Downloads/flower_original.jpeg')  


height, width, _ = image.shape  

x_start, y_start = 0, 0  


region = image[y_start:y_start+3, x_start:x_start+3]


print(region)
print(image[20][20])