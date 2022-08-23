from color_analysis import *

if __name__ == '__main__':

    img_result, roi_list, img_src = thres_segmentation('black_img/sample/camera16.png', 25, 20, 8)

    # for i in range(len(roi_list)):
    #
    #     roi = get_single_roi(img_src,roi_list[i])
    #     crop_roi = roi_crop(roi,80,120)
    # show(img_src,"contour",True)