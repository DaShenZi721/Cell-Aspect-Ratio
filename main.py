import numpy as np
import cv2
from skimage import measure
from skimage.color import rgb2gray
from skimage.morphology import convex_hull_image
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import sys

print(cv2.__version__)

drawing = False  # true if mouse is pressed
find_contour = False
pt1_x, pt1_y = None, None
big_contour_copy = None
scale = 1.0

def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),-1)
    return cv_img

def mouse_event(event, x, y, flags, param):
    global pt1_x, pt1_y, drawing, img, find_contour, big_contour_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
        cv2.imshow('image', img)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=5)
            pt1_x, pt1_y = x, y
            cv2.imshow('image', img)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (pt1_x, pt1_y), (x, y), color=(255, 255, 255), thickness=5)
        cv2.imshow('image', img)

    if event == cv2.EVENT_RBUTTONDOWN:
        if find_contour == False:
            lower = (255, 255, 255)
            upper = (255, 255, 255)
            thresh = cv2.inRange(img, lower, upper)
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)

            mask = np.zeros_like(img)
            cv2.drawContours(mask, [big_contour], 0, (255, 255, 255), -1)
            img = cv2.bitwise_and(img, mask)
            img[np.where((img == [255, 255, 255]).all(axis=2))] = [0, 0, 0]

            level = 0.1
            gray_img = rgb2gray(img)
            contours = measure.find_contours(gray_img, level)
            big_contour = max(contours, key=lambda x: x.shape[0])
            big_contour_copy = big_contour.copy()
            big_contour = np.expand_dims(big_contour, axis=1).astype(int)[:, :, ::-1]
            cv2.drawContours(img, [big_contour], 0, (255, 0, 255), thickness=2)
            find_contour = True

        # print("x:", x, " y:", y, img[y, x, :])

        poly = Polygon(big_contour_copy)
        box = poly.minimum_rotated_rectangle
        x, y = box.exterior.coords.xy
        pts = np.stack([x,y]).astype(int).T[:, ::-1]
        cv2.polylines(img, [pts], True, (255, 255, 255), 2)
        edge_length = (Point(x[0], y[0]).distance(Point(x[1], y[1])), Point(x[1], y[1]).distance(Point(x[2], y[2])))
        length = max(edge_length)
        width = min(edge_length)
        print('Aspect Ratio: %f' % (length / width))
        cv2.imshow('image', img)


if __name__ == '__main__':
    # filename = sys.argv[1]
    filename = r'example.tif'
    print(filename)
    # img = cv2.imread(filename)
    img = cv_imread(filename)
    origin_img = img.copy()
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(h / 3), int(w / 3)))

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_event)
    cv2.imshow('image', img)

    while True:
        # print(cv2.waitKey(2))
        # S
        if cv2.waitKey(2) == 115:
            scale = input('请输入放缩倍数：')
            scale = float(scale)
            img = origin_img.copy()
            img = cv2.resize(img, (int(h*scale/3), int(w*scale/3)))
            cv2.imshow('image', img)
        # A
        if cv2.waitKey(2) == 97:
            drawing = False  # true if mouse is pressed
            find_contour = False
            pt1_x, pt1_y = None, None
            big_contour_copy = None
            img = origin_img.copy()
            img = cv2.resize(img, (int(h*scale/3), int(w*scale/3)))
            cv2.imshow('image', img)
        if cv2.waitKey(2) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
