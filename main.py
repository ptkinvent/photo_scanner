#!/usr/bin/python3

import cv2
import numpy as np

"""
Resource: https://learnopencv.com/automatic-document-scanner-using-opencv/
"""

# Read the original image
img_orig = cv2.imread('data/DSC_5122.JPG')

# Resize smaller
scale_percent = 10
width = int(img_orig.shape[1] * scale_percent/100)
height = int(img_orig.shape[0] * scale_percent/100)
dim = (width, height)
img = cv2.resize(img_orig, dim, interpolation=cv2.INTER_AREA)

# Display resized image
cv2.imshow('Original', img)
cv2.waitKey(0)

# Blur the image for better edge detection
img = cv2.GaussianBlur(img, (3,3), 0)

# Select only the orange pixels from the image
light_orange = (12, 100, 0)
dark_orange = (25, 255, 255)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
mask = cv2.inRange(img_hsv, light_orange, dark_orange)
img = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('Filtered', img)
cv2.waitKey(0)

# Open and close the image repeatedly to blank the photograph
kernel = np.ones((5,5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=6)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=20)
cv2.imshow('Closed', img)
cv2.waitKey(0)

# Convert all black pixels to white and other pixels to black
black_pixels = np.where(
    (img[:, :, 0] == 0) &
    (img[:, :, 1] == 0) &
    (img[:, :, 2] == 0)
)
white_pixels = np.where(
    (img[:, :, 0] > 0) |
    (img[:, :, 1] > 0) |
    (img[:, :, 2] > 0)
)
img[black_pixels] = [255, 255, 255]
img[white_pixels] = [0, 0, 0]

# GrabCut
mask = np.zeros(img.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
rect = (20, 20, img.shape[1]-20, img.shape[0]-20)
cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img = img*mask2[:,:,np.newaxis]
cv2.imshow('GrabCut', img)
cv2.waitKey(0)

# Canny Edge Detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(image=gray, threshold1=0, threshold2=200) # Canny Edge Detection
canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
cv2.imshow('Edges', canny)
cv2.waitKey(0)

# Contour detection
con = np.zeros_like(img)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

con = np.zeros_like(img)
for c in page:
    epsilon = 0.02*cv2.arcLength(c, True)
    corners = cv2.approxPolyDP(c, epsilon, True)
    if len(corners) == 4:
        break
cv2.drawContours(con, c, -1, (0, 255, 255), 3)
cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
corners = sorted(np.concatenate(corners).tolist())

for index, c in enumerate(corners):
    character = chr(65+index)
    cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

cv2.imshow('Contours', con)
cv2.waitKey(0)

cv2.destroyAllWindows()
