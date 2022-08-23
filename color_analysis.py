from area_extract import *
import numpy as np
import cv2 as cv


def roi_crop(roi,nw,nh):
    """
    :param roi: 获得的单个显示屏区域
    :param nw: 中心裁剪的宽度
    :param nh: 中心裁剪的高度
    :return: 裁剪后的区域
    """
    oh, ow = roi.shape[0], roi.shape[1]
    # roi = cv.resize(roi,(nw,nh))
    c_x = ow // 2
    c_y = oh // 2
    roi = roi[c_y-(nh//4):c_y+(nh//4),c_x-(nw//4):c_x+(nw//4),:]
    # show(roi,"resize_roi",True)
    return roi

def analysis(img_src):
    """
    :param img_src: 原图
    :return:
    """
    img = cv.cvtColor(img_src,cv.COLOR_BGR2HSV)
    average_H = np.sum(img[:,:,0])/(img.shape[0]*img.shape[1])
    average_H = round(average_H,2)
    # 120.00是观察得到的经验值
    if 120.00 < average_H:
        print("显示屏故障")
    return average_H



if __name__ == '__main__':
    img_list = read_picture("black_img/sample")
    for i in range(len(img_list)):
        average_H_list = []
        img_result, roi_list, img_src = thres_segmentation(img_list[i], 25, 20, 10)

        img = cv.cvtColor(img_src, cv.COLOR_BGR2HSV)
        # show(img[:,:,0], "H", True, img_src)

        for j in range(len(roi_list)):
            roi = get_single_roi(img_src, roi_list[j])
            # show(roi, "roi", True)
            roi_c = roi_crop(roi,400,600)
            average_H = analysis(roi_c)
            average_H_list.append(average_H)
            text_roi = cv.putText(roi, "%2.f" % average_H, \
                (100, 100), cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 5)
            show(text_roi,"text_roi",True)
            x,y,w,h = roi_list[j]
            img_src[y:y+h,x:x+w,:] = text_roi

        print(average_H_list)