#!/usr/bin/python3

'''
Reference: https://realpython.com/python-opencv-color-spaces/
'''

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

img = cv2.imread('./data/DSC_5122.JPG')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize smaller
scale_percent = 5
width = int(img.shape[1] * scale_percent/100)
height = int(img.shape[0] * scale_percent/100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

r, g, b = cv2.split(img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
# plt.show()



hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()


light_orange = (12, 0, 0)
dark_orange = (25, 255, 255)

mask = cv2.inRange(hsv_img, light_orange, dark_orange)
result = cv2.bitwise_and(img, img, mask=mask)
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()
