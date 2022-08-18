import cv2 as cv
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import copy

def read_picture(images_src):
    img_list = os.listdir(images_src)
    for idx, _ in enumerate(img_list):
        # img_list[idx] = f"{os.getcwd()}" + "/img/" + img_list[idx]
        img_list[idx] = images_src + '/' + img_list[idx]
    return img_list


# def thres_segmentation(img_list):
#     th_img_list = []
#     mask_list = []
#     th_list = []
#     for idx, val in enumerate(img_list):
#         img_src = cv.imread(img_list[idx])
#         img_shape = img_src.shape
#         img_gray = cv.cvtColor(img_src,cv.COLOR_BGR2GRAY)
#
#         img = cv.medianBlur(img_gray,5)
#         th = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                              cv.THRESH_BINARY,61,2)
#         # blur = cv.GaussianBlur(img_gray,(5,5),0)
#         # ret, th = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#         # ret, th = cv.threshold(blur,127,255,cv.THRESH_BINARY)
#         th_list.append(th)
#         # 去除噪点
#         # kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#         # img_result = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)
#
#         # 制作掩码图
#         contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#         img_mask = np.zeros(img_shape,np.uint8)
#         cv.drawContours(img_mask,contours,-1,(255,255,255),3)
#         mask_list.append(img_mask)
#
#         # 提取前景
#         img_result = cv.bitwise_and(img_src,img_mask)
#         th_img_list.append(img_result)
#
#
#     return th_img_list,mask_list,th_list


# 非黑色背景
def thres_segmentation(img,open_kernel_size,window_num):
    img_src = cv.imread(img)
    img_hsv = cv.cvtColor(img_src,cv.COLOR_BGR2HSV)
    lower = np.array([0, 0, 127])
    upper = np.array([179, 255, 255])

    mask = cv.inRange(img_hsv,lower,upper)
    ret, th = cv.threshold(mask,127,255,cv.THRESH_BINARY)
    # show(th,name="th")
    # 去噪
    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
    img_result = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_open)
    # show(img_result,name="img_result")

    # kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # img_result = cv.morphologyEx(img_result, cv.MORPH_CLOSE, kernel_close)
    # edge = cv.Canny(img_result)
    # img_result = cv.cvtColor(img_result,cv.COLOR_GRAY2BGR)
    # img_result = cv.bitwise_and(img_src, img_result)
    roi_list = get_roi(img_src,img_result, window_num)

    return img_result,roi_list,img_src


def get_roi(original_img,binary,window_num):
    """
    :param original_img: 原始图像
    :param binary: 经过去噪后的图像
    :param window_num: 用来对生成最小面积的参数（一般大于显示屏的个数）
    :return: 返回roi的列表
    """
    contours, _ = cv.findContours(
        binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    roi_list = []
    area_list = []
    # count = 8
    # min_area = original_img.shape[0] * original_img.shape[1] / count / 4
    for cnt in range(len(contours)):
        area = cv.contourArea(contours[cnt])
        area_list.append(area)
    area_list_copy = copy.deepcopy(area_list)
    area_list_copy.sort(reverse=True)
    min_area = sum(area_list_copy[:window_num]) / window_num
    # 判断矩形区域
    for cnt in range(len(contours)):
        area = area_list[cnt]
        # 判断提取所需的轮廓，经验值需要调试获取
        if area > min_area:
            # 获取外接矩形的值
            x, y, w, h = cv.boundingRect(contours[cnt])
            roi_list.append((x, y, w, h))
            cv.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv.drawContours(original_img, [contours[cnt]], 0, (255, 0, 255), 2)

            # 内接矩形
            # rect = order_points(contours[cnt].reshape(contours[cnt].shape[0],2))
            # print(rect)
            # xs = [i[0] for i in rect]
            # ys = [i[1] for i in rect]
            # xs.sort()
            # ys.sort()
            # 内接矩形的坐标为
            # print(xs[1], xs[2], ys[1], ys[2])
            # cv.rectangle(original_img, (int(xs[0]), int(ys[0])), (int(xs[3]), int(ys[3])), (255, 0, 0), 3)
            # cv.drawContours(original_img, [contours[cnt]], 0, (0, 0, 255), 2)

    # show(original_img,"orginal",True)
    # for i in range(len(roi_list)):
    #     x, y, w, h = roi_list[i]
    #     roi = original_img[y:y+h,x:x+w,:]
    #     show(roi,f"roi{i}",True)

    return roi_list


def get_single_roi(img,roi_list):
    """
    :param img: 原图
    :param roi_list: 前景对象坐标列表
    """
    img_src = cv.imread(img)
    # single_img_rois = []
    # for i in range(len(roi_list)):
    #     x, y, w, h = roi_list[i]
    #     roi = img_src[y:y+h,x:x+w,:]
    #     # show(roi,f"roi{i+1}",True)
    #     single_img_rois.append(roi)
    # return single_img_rois

    x, y, w, h = roi_list
    roi = img_src[y:y+h,x:x+w,:]
    return roi

# 用于获取内接矩形四个点的函数
def order_points(pts):
    # pts为轮廓坐标
    # 列表中存储元素分别为左上角，右上角，右下角和左下角
    rect = np.zeros((4, 2), dtype = "float32")
    # 左上角的点具有最小的和，而右下角的点具有最大的和
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 计算点之间的差值
    # 右上角的点具有最小的差值,
    # 左下角的点具有最大的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # 返回排序坐标(依次为左上右上右下左下)
    return rect


def show(img,name,waitkey=False):
    # img = cv.resize(img, (500, 400))
    cv.imshow(name, img)
    if waitkey:
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == '__main__':
    img_list = read_picture("./img")

    th_img_list = []
    for i in range(len(img_list)):
        img_result,roi_list = thres_segmentation(img_list[i],30,10)
        print(type(img_list[i]))
        # th_img_list.append(img_result)
        for j in range(len(roi_list)):
            roi = get_single_roi(img_list[i],roi_list[j])
            show(roi,"roi",True)

    # for i in range(4):
    #     plt.subplot(2, 2, i + 1), plt.imshow(th_img_list[i], 'gray')
    #     plt.title(i)
    #     plt.xticks([]), plt.yticks([])
    # plt.show()

