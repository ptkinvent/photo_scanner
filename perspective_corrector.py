#!/usr/bin/python3

import cv2
import numpy as np


class PerspectiveCorrector:
    """
    Given an image and 4 coordinates, this class corrects perspective warp
    Resource: https://learnopencv.com/automatic-document-scanner-using-opencv/
    """

    def __init__(self):
        pass

    @staticmethod
    def _dist(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def _find_dest_coords(pts):
        (tl, tr, br, bl) = pts

        widthA = PerspectiveCorrector._dist(br, bl)
        widthB = PerspectiveCorrector._dist(tr, tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = PerspectiveCorrector._dist(tr, br)
        heightB = PerspectiveCorrector._dist(tl, bl)
        maxHeight = max(int(heightA), int(heightB))

        # Final destination coord
        return [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]

    @staticmethod
    def correct_perspective(img, corners):
        # Getting the homography
        destination_corners = PerspectiveCorrector._find_dest_coords(corners)
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
        # Perspective transform using homography
        final = cv2.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)
        return final
