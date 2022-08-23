# 程序入口
from area_extract import *
from color_analysis import *

if __name__ == '__main__':

    # 摄像头实时版本
    for cn_num in range(0, 5):
        cap = cv.VideoCapture(cn_num)
        if cap.isOpened():
            break
    while True:
        ret, frame = cap.read()
        # frame = cv.imread("black_img/sample/camera5.png")
        img_result, roi_list, = thres_segmentation(frame,25,20,8)
        for j in range(len(roi_list)):
            roi = get_single_roi(frame,roi_list[j])
            roi_c = roi_crop(roi, 80, 120)
            average_H = analysis(roi_c)
            text_roi = cv.putText(roi, "%2.f" % average_H, \
                                  (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            x, y, w, h = roi_list[j]
            frame[y:y + h, x:x + w, :] = text_roi
        cv.imshow("camera",frame)
        if cv.waitKey(25) & 0xFF == 27:
            break


    cap.release()
    cv.destroyAllWindows()

    # 本地测试图片测试版本
    # img_list = read_picture("black_img/sample")
    # th_img_list = []
    # average_H_list = []
    # for i in range(len(img_list)):
    #     img_src = cv.imread(img_list[i])
    #     img_result, roi_list, = thres_segmentation(img_src,25,20,8)
    #
    #     img = cv.cvtColor(img_src, cv.COLOR_BGR2HSV)
    #     # show(img[:, :, 0], " H", True, img_src)
    #
    #     for j in range(len(roi_list)):
    #         roi = get_single_roi(img_src,roi_list[j])
    #         # print(roi.shape)
    #         # show(roi,"roi",True)
    #         roi_c = roi_crop(roi, 80, 120)
    #         average_H = analysis(roi_c)
    #         average_H_list.append(average_H)
    #         text_roi = cv.putText(roi, "%2.f" % average_H, \
    #                               (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
    #         # show(text_roi, "text_roi", True)
    #         x, y, w, h = roi_list[j]
    #         img_src[y:y + h, x:x + w, :] = text_roi
    #     show(img_src,"text_img",True)
    #     print(average_H_list)