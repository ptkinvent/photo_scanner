#!/usr/bin/python3

import cv2
import numpy as np


class CornerDetector:
    """
    Automatically detects 4 corners within an image
    Resource: https://learnopencv.com/automatic-document-scanner-using-opencv/
    """

    def __init__(self):
        pass

    @staticmethod
    def _order_points(pts):
        '''Rearrange coordinates to order:
        top-left, top-right, bottom-right, bottom-left'''
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # Return the ordered coordinates.
        return rect.astype('int').tolist()

    @staticmethod
    def detect_corners(img, debug=False):
        # Blur the image for better edge detection
        img = cv2.GaussianBlur(img, (3,3), 0)

        # Select only the orange pixels from the image
        light_orange = (12, 100, 0)
        dark_orange = (25, 255, 255)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, light_orange, dark_orange)
        img = cv2.bitwise_and(img, img, mask=mask)
        if debug:
            cv2.imshow('Filtered', img)
            cv2.waitKey(0)

        # Open and close the image repeatedly to blank the photograph
        kernel = np.ones((5,5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=6)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=20)
        if debug:
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
        if debug:
            cv2.imshow('GrabCut', img)
            cv2.waitKey(0)

        # Canny Edge Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        canny = cv2.Canny(image=gray, threshold1=0, threshold2=200) # Canny Edge Detection
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        if debug:
            cv2.imshow('Edges', canny)
            cv2.waitKey(0)

        # Contour detection
        con = np.zeros_like(img)
        contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        con = cv2.drawContours(con, page, -1, (0, 255, 255), 3)

        for c in page:
            epsilon = 0.02*cv2.arcLength(c, True)
            corners = cv2.approxPolyDP(c, epsilon, True)
            if len(corners) == 4:
                break
        cv2.drawContours(con, c, -1, (0, 255, 255), 3)
        cv2.drawContours(con, corners, -1, (0, 255, 0), 10)
        corners = sorted(np.concatenate(corners).tolist())
        corners = CornerDetector._order_points(corners)

        for index, c in enumerate(corners):
            character = chr(65+index)
            cv2.putText(con, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        if debug:
            cv2.imshow('Contours', con)
            cv2.waitKey(0)

        return corners
