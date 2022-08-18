import cv2 as cv
import numpy as np
import os


def read_picture(images_src):
    img_list = os.listdir(images_src)
    for idx, _ in enumerate(img_list):
        img_list[idx] = images_src + '/' + img_list[idx]
    print(img_list)
    return img_list

# 滑动条的回调函数，获取滑动条位置处的值
def empty(a):
    h_min = cv.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    return h_min, h_max, s_min, s_max, v_min, v_max

path = read_picture("./img")
# 创建一个窗口，放置6个滑动条
cv.namedWindow("TrackBars", cv.WINDOW_AUTOSIZE)
cv.resizeWindow("TrackBars", 640, 240)
cv.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv.createTrackbar("Hue Max", "TrackBars", 19, 179, empty)
cv.createTrackbar("Sat Min", "TrackBars", 110, 255, empty)
cv.createTrackbar("Sat Max", "TrackBars", 240, 255, empty)
cv.createTrackbar("Val Min", "TrackBars", 153, 255, empty)
cv.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv.imread(path[0])
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 调用回调函数，获取滑动条的值
    h_min, h_max, s_min, s_max, v_min, v_max = empty(0)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    # 获得指定颜色范围内的掩码
    mask = cv.inRange(imgHSV, lower, upper)
    # 对原图图像进行按位与的操作，掩码区域保留
    imgResult = cv.bitwise_and(img, img, mask=mask)


    cv.imshow("Mask", mask)
    cv.imshow("Result", imgResult)
    cv.resizeWindow("Mask",640,640)
    cv.resizeWindow("Result", 640, 640)
    cv.waitKey(1)