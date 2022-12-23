#!/usr/bin/python3

import cv2
import numpy as np

"""
Resource: https://learnopencv.com/automatic-document-scanner-using-opencv/
"""

# Read the original image
img = cv2.imread('data/DSC_5122.JPG')

# Resize smaller
scale_percent = 10
width = int(img.shape[1] * scale_percent/100)
height = int(img.shape[0] * scale_percent/100)
dim = (width, height)
img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# Display resized image
cv2.imshow('Original', img_resized)
cv2.waitKey(0)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_resized, (3,3), 0)

# Close the image repeatedly to blank the photograph
kernel = np.ones((5,5), np.uint8)
img_closed = cv2.morphologyEx(img_blur, cv2.MORPH_CLOSE, kernel, iterations=3)
# cv2.imshow('Closed', img_closed)
# cv2.waitKey(0)

mask = np.zeros(img_resized.shape[:2], np.uint8)
bgModel = np.zeros((1, 65), np.float64)
fgModel = np.zeros((1, 65), np.float64)
rect = (20, 20, img_resized.shape[1]-20, img_resized.shape[0]-20)
cv2.grabCut(img_closed, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
img_gc = img_closed*mask2[:,:,np.newaxis]
# cv2.imshow('GrabCut', img_gc)
# cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img_gc, cv2.COLOR_BGR2GRAY)

# Gaussian blur
img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)

# Canny Edge Detection
img_edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=200) # Canny Edge Detection

# Dilate
img_dilate = cv2.dilate(img_edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
cv2.imshow('GrabCut', img_dilate)
cv2.waitKey(0)

# Contour detection
con = np.zeros_like(img_resized)
contours, hierarchy = cv2.findContours(img_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

con = np.zeros_like(img_resized)
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
