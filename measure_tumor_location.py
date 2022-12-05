import cv2
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import cv2
import math

import torch
print(torch.cuda.is_available())


HEIGHT = 128
WIDTH = 128
IMG_CHANNELS = 3

img = cv2.imread('image1.jpg')
img = cv2.resize(img, (HEIGHT, WIDTH))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
img = np.array(img)
plt.imshow(img)
plt.show()

# print("img: ", img)

mask_synthesis = np.zeros((HEIGHT, WIDTH, IMG_CHANNELS), dtype=np.uint8)
# Convert BGR to HSV
mask_HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


low_bound = np.array([21, 27, 121])
up_bound = np.array([70, 80, 200])
temp = cv2.inRange(mask_HSV, low_bound, up_bound)
for i in range(HEIGHT):
    for j in range(WIDTH):
        if temp[i][j] == 255:
            mask_synthesis[i][j] = np.array([255, 255, 255])

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.imshow(np.array(img))
f.add_subplot(1, 2, 2)
plt.imshow(np.array(mask_synthesis))
plt.show(block=True)

# print(mask_synthesis)

print("image shape", img.shape)
print("\n")

interested_in_color = [0, 0, 0]  # RGB

all_pixels = HEIGHT * WIDTH

interested_in_color = set(interested_in_color)
roi_pixels = 0
for i in mask_synthesis:
    for pixel in i:
        pixel = set(pixel)
        if interested_in_color != pixel:
            roi_pixels += 1

# print("roi_pixels", roi_pixels)
# print("all_pixels", all_pixels)

roi_percentage = round(roi_pixels / all_pixels * 100, 2)
print("The ROI of the tumor has size ", roi_percentage, "% of the whole x-ray")


# Load image
im = mask_synthesis

# Convert to grayscale and threshold
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 1, 255, 0)

# Find contours, draw on image and save
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0, 255, 0), 3)
cv2.imwrite('resultcontour.png', im)

# Show user what we found
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(x), int(y))
    radius = int(radius)
    area = math.pi * (radius ^ 2)
    print('\nContour: centre {},{}, radius {}'.format(x, y, radius))
    print('\nThe total Area of the tumor is {} (pixels for the time being)'.format(round(area)))


# Find the quadradic
yUpperLower_quadradic = HEIGHT/2
xLeftRight_quadradic = WIDTH/2


print('\n')

if x < xLeftRight_quadradic and y < yUpperLower_quadradic:
    print(" The tumor belongs to upper left quadrant.")
elif x > xLeftRight_quadradic and y < yUpperLower_quadradic:
    print(" The tumor belongs to upper right quadrant.")
elif x < xLeftRight_quadradic and y > yUpperLower_quadradic:
    print(" The tumor belongs to lower left quadrant.")
else:
    print(" The tumor belongs to lower right quadrant.")

