import cv2
import numpy as np

image = cv2.imread('/Users/adityamaindan/Downloads/flower_original.jpeg')  

image_data = image.reshape(-1, 3)

first_pixel_rgb = image_data[0]
print("RGB values of the first pixel:", first_pixel_rgb)

import pandas as pd


df = pd.DataFrame(image_data, columns=['R', 'G', 'B'])

df.to_csv('rgb_values.csv', index=False)

