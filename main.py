# 程序入口
from area_extract import *
from color_analysis import *

if __name__ == '__main__':
    img_list = read_picture("black_img/sample")
    th_img_list = []
    average_H_list = []
    for i in range(len(img_list)):
        print(img_list[i])
        img_result, roi_list, img_src = thres_segmentation(img_list[i],25,20,8)

        img = cv.cvtColor(img_src, cv.COLOR_BGR2HSV)
        # show(img[:, :, 0], " H", True, img_src)

        for j in range(len(roi_list)):
            roi = get_single_roi(img_src,roi_list[j])
            # print(roi.shape)
            # show(roi,"roi",True)
            roi_c = roi_crop(roi, 80, 120)
            average_H = analysis(roi_c)
            average_H_list.append(average_H)
            text_roi = cv.putText(roi, "%2.f" % average_H, \
                                  (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            # show(text_roi, "text_roi", True)
            x, y, w, h = roi_list[j]
            img_src[y:y + h, x:x + w, :] = text_roi
        show(img_src,"text_img",True)
        print(average_H_list)