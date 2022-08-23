import cv2 as cv
import os
import numpy as np
import copy


def read_picture(images_src):
    img_list = os.listdir(images_src)
    for idx, _ in enumerate(img_list):
        img_list[idx] = images_src + '/' + img_list[idx]
    return img_list


def thres_segmentation(img_path,open_kernel_size,close_kernel_size,window_num):
    img_src = cv.imread(img_path)
    img_gray = cv.cvtColor(img_src,cv.COLOR_BGR2GRAY)
    img_hsv = cv.cvtColor(img_src,cv.COLOR_BGR2HSV)

    ret1, th_Binary = cv.threshold(img_gray,205,255,cv.THRESH_BINARY)
    ret2, th_Ostu = cv.threshold(img_hsv[:,:,2],127,255, cv.THRESH_OTSU)

    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
    img_hsv_result = cv.morphologyEx(th_Ostu, cv.MORPH_OPEN, kernel_open)
    kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
    img_gray_result = cv.morphologyEx(th_Binary,cv.MORPH_OPEN,kernel_open)


    # 解决两块显示屏挨在一起过亮的问题
    contours_thres, _ = cv.findContours(
        img_gray_result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_ostu, _ = cv.findContours(
        img_hsv_result, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # 解决难以发现的小轮廓导致下面判断轮廓数量数显问题
    for i in range(len(contours_ostu)):
        area = cv.contourArea(contours_ostu[i])
        if area < 20:
            delete_idx = i
            contours_ostu = contours_ostu[:delete_idx] + contours_ostu[delete_idx+1:]
    # print(len(contours_ostu))
    # show(img_gray_result, "th")
    # show(img_hsv_result, "th_ostu", )
    # show(img_src, "src", True)
    if len(contours_ostu) >= len(contours_thres):
        contours = contours_ostu
        img_result = img_hsv_result
    else:
        contours = contours_thres
        img_result = img_gray_result
    roi_list = get_roi(img_src, contours, window_num)

    return img_result,roi_list,img_src


# 黑色背景
# def thres_segmentation(img_path,open_kernel_size,close_kernel_size,window_num):
#     img_src = cv.imread(img_path)
#
#     img_hsv = cv.cvtColor(img_src,cv.COLOR_BGR2HSV)
#     print(img_hsv.shape)
#     # lower与upper用于调节hsv中颜色区间的
#     # lower = np.array([0, 0, 127])
#     # upper = np.array([179, 255, 255])
#     # mask = cv.inRange(img_hsv,lower,upper)
#     # ret, th = cv.threshold(mask,127,255,cv.THRESH_BINARY)
#     show(img_hsv[:,:,2],"V",True)
#     ret, th = cv.threshold(img_hsv[:,:,2],0,255, cv.THRESH_OTSU)
#
#     show(th,"th",True,stack_img=img_src)
#
#     # 去噪
#     kernel_open = cv.getStructuringElement(cv.MORPH_RECT, (open_kernel_size, open_kernel_size))
#     img_result = cv.morphologyEx(th, cv.MORPH_OPEN, kernel_open)
#     # show(img_result,"open_and_binary",True,stack_img=resized_img_src)
#
#     kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (close_kernel_size, close_kernel_size))
#     img_result = cv.morphologyEx(img_result, cv.MORPH_CLOSE, kernel_close)
#     show(img_result,"close_and_binary",True,img_src)
#     # edge = cv.Canny(img_result)
#     # img_result = cv.cvtColor(img_result,cv.COLOR_GRAY2BGR)
#     # img_result = cv.bitwise_and(img_src, img_result)
#     roi_list = get_roi(img_src,img_result, window_num)
#
#     return img_result,roi_list,img_src


def get_roi(original_img, contours, window_num):
    """
    :param original_img: 原始图像
    :param binary: 经过去噪后的图像
    :param window_num: 用来对生成最小面积的参数（一般大于显示屏的个数）
    :return: 返回roi的列表
    """

    roi_list = []
    area_list = []

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
    x, y, w, h = roi_list
    roi = img[y:y+h,x:x+w,:]
    return roi


def show(img,name,waitkey=False,stack_img=None):
    """
    统一用来显式图像的函数
    :param img:一般来讲是二值图
    :param name: 窗口名
    :param waitkey:
    :param stack_img: 一般来讲是原图
    :return: none
    """
    if stack_img is not None:
        # 单通道变为三通道方便拼接显示
        img = cv.merge([img,img,img])
        img = np.hstack([img,stack_img])
    cv.namedWindow(name,cv.WINDOW_NORMAL)
    cv.imshow(name, img)
    if waitkey:
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == '__main__':
    img_list = read_picture("black_img/sample")

    th_img_list = []
    for i in range(len(img_list)):
        img_result,roi_list,img_src = thres_segmentation(img_list[i],20,20,8)
        # th_img_list.append(img_result)
        for j in range(len(roi_list)):
            roi = get_single_roi(img_src,roi_list[j])
            # show(roi,"roi",True)


