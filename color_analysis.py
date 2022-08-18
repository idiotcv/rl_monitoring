from area_extract import *
import numpy as np
import cv2 as cv


def roi_crop(roi,nw,nh):
    ow, oh = roi.shape[0], roi.shape[1]
    # roi = cv.resize(roi,(nw,nh))
    c_x = ow // 2
    c_y = oh // 2
    roi = roi[c_y-(nh//4):c_y+(nh//4),c_x-(nw//4):c_x+(nw//4),:]
    # show(roi,"resize_roi",True)
    return roi

def analysis(img_src):

    img = cv.cvtColor(img_src,cv.COLOR_BGR2HSV)
    # average_H = sum(img[:][:][0])/(img.shape[0]*img.shape[1])
    # H,S,V = cv.split(img)
    average_H = np.sum(img[:,:,0])/(img.shape[0]*img.shape[1])
    # if 280 < average_H < 320:
    #     print("显示屏故障")
    # cv.putText(img_src,"average_H:%2.f"%average_H,(100, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return round(average_H,2)



if __name__ == '__main__':
    img_list = read_picture("./img")
    for i in range(len(img_list)):
        average_H_list = []
        img_result, roi_list, img_src = thres_segmentation(img_list[i], 30, 10)

        img = cv.cvtColor(img_src, cv.COLOR_BGR2HSV)
        show(cv.resize(img[:, :, 0], (600, 800)), "H", )

        for j in range(len(roi_list)):
            roi = get_single_roi(img_list[i], roi_list[j])
            print(roi.shape)
            # show(roi, "roi", True)
            roi_c = roi_crop(roi,400,600)
            average_H = analysis(roi_c)
            average_H_list.append(average_H)
            text_roi = cv.putText(roi, "%2.f" % average_H, \
                (100, 100), cv.FONT_HERSHEY_COMPLEX, 5, (0, 0, 255), 5)
            # show(text_roi,"text_roi",True)
            x,y,w,h = roi_list[j]
            img_src[y:y+h,x:x+w,:] = text_roi


        show(cv.resize(img_src,(600,800)),"test",True)

        # for cnt in range(len(average_H_list)):
        #     cv.putText(img_src, "average_H:%2.f" % average_H_list[cnt], \
        #                (300, 300), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        #     if cnt > 0:
        #         cv.putText(img_src, "average_H:%2.f" % average_H_list[cnt], \
        #             (300, 300), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # show(img_src,"img",True)
        print(average_H_list)