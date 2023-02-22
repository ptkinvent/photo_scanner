#!/usr/bin/python3

'''
Reference: https://realpython.com/python-opencv-color-spaces/
'''

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors

if len(sys.argv) != 2:
    print('Usage: python3 main.py <fname>')
    exit(0)
fname = sys.argv[1]
img = cv2.imread(fname)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize smaller
scale = 5/100
width = int(img.shape[1] * scale)
height = int(img.shape[0] * scale)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Plot RGB
r, g, b = cv2.split(img)
fig_rgb = plt.figure()
axis = fig_rgb.add_subplot(1, 1, 1, projection="3d")
axis.set_title("RGB")

pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")

# Plot HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv_img)
fig_hsv = plt.figure()
axis = fig_hsv.add_subplot(1, 1, 1, projection="3d")
axis.set_title("HSV")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()
