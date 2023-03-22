#!/usr/bin/python3

import sys
import cv2
import numpy as np
from corner_detector import CornerDetector

"""
Resource: https://stackoverflow.com/questions/42369536/drag-mouse-to-draw-a-line-and-get-cordinates-of-end-points-of-line-in-opencv-pyt
"""

drag = False


def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_dest_coords(pts):
    (tl, tr, br, bl) = pts

    widthA = dist(br, bl)
    widthB = dist(tr, tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = dist(tr, br)
    heightB = dist(tl, bl)
    maxHeight = max(int(heightA), int(heightB))

    # Final destination coord
    return [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 mouse_handler.py <fname>')

    img_orig = cv2.imread(sys.argv[1])

    # Resize smaller
    scale = 1/10
    width = int(img_orig.shape[1] * scale)
    height = int(img_orig.shape[0] * scale)
    dim = (width, height)
    img = cv2.resize(img_orig, dim, interpolation=cv2.INTER_AREA)

    # Draw 4 handles
    tl_handle = (50, 50)
    tr_handle = (width-50, 50)
    br_handle = (width-50, height-50)
    bl_handle = (50, height-50)
    handles = [tl_handle, tr_handle, br_handle, bl_handle]
    color_magenta = (255, 0, 255)

    cv2.imshow('image', img)

    # Enable dragging of handles
    def mouse_handler(event, x, y, flags, data):
        global drag
        if event == cv2.EVENT_LBUTTONDOWN:
            drag = True

        if event == cv2.EVENT_LBUTTONUP and drag:
            drag = False

        if drag and event == cv2.EVENT_MOUSEMOVE:
            # Find the closest handle (manhattan distance is good enough)
            closest_handle_idx = 0
            for i, handle in enumerate(handles):
                if dist((x, y), handle) < dist((x, y), handles[closest_handle_idx]):
                    closest_handle_idx = i
            handles[closest_handle_idx] = (x, y)

    cv2.setMouseCallback('image', mouse_handler)

    while True:
        img_annotated = cv2.resize(img_orig, dim, interpolation=cv2.INTER_AREA)
        for i in range(4):
            cv2.circle(img_annotated, handles[i], radius=10, color=color_magenta, thickness=5)
            cv2.line(img_annotated, handles[i-1], handles[i], color_magenta, thickness=1)
        cv2.imshow('image', img_annotated)
        k = cv2.waitKey(1)

        # Spacebar to output image
        if k == 32:
            # Getting the homography
            corners = [(x/scale, y/scale) for (x, y) in handles]
            destination_corners = find_dest_coords(corners)
            M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
            # Perspective transform using homography
            final = cv2.warpPerspective(img_orig, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LINEAR)

            # Output
            print('Outputting...')
            cv2.imwrite('final.jpg', final)
            cv2.destroyAllWindows()
            break

        # 'A' key to auto-detect corners
        elif k == ord('a'):
            corner_detector = CornerDetector()
            handles = corner_detector.detect(img)

        # Esc key to exit
        elif k == 27:
            cv2.destroyAllWindows()
            break


main()
