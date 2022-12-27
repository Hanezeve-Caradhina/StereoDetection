# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import stereoconfigcopy
import math
from PIL import Image
import time
    

if __name__ == '__main__':
    time_start = time.time()
    # 读取MiddleBurry数据集的图片
    iml = cv2.imread('biaoding/zuo/left_2000.jpg', 1)  # 左图
    imr = cv2.imread('biaoding/you/right_2000.jpg', 1)  # 右图
    if (iml is None) or (imr is None):
        print("Error: Images are empty, please check your image's path!")
        sys.exit(0)
    #height, width = iml.shape[0:2]

    #iml_new = cv2.resize(iml, (640, 360))
    #imr_new = cv2.resize(imr, (640, 360))
    #cv2.imwrite("./SaveImage/test1.jpg", iml_new)
    #cv2.imwrite("./SaveImage/test2.jpg", imr_new)
 
    # 读取相机内参和外参
    # 使用之前先将标定得到的内外参数填写到stereoconfig.py中的StereoCamera类中
    config = stereoconfigcopy.stereoCamera()
    config.__init__()

    imgL = cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
    imgR = cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('res_image/disaprity.png', imgL)
    # 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
    height, width = iml.shape[0:2]
    print(height, width)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(config.cam_matrix_left, 
                                                                      config.distortion_l,
                                                                      config.cam_matrix_right, 
                                                                      config.distortion_r, 
                                                                      (width, height), 
                                                                      config.R,
                                                                      config.T)
    left_map1, left_map2 = cv2.initUndistortRectifyMap(config.cam_matrix_left, config.distortion_l, R1, P1, (width, height), cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(config.cam_matrix_right, config.distortion_r, R2, P2, (width, height), cv2.CV_16SC2)

    # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    #imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR).astype(np.uint8)
    #imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    

    blockSize = 8
    img_channels = 1
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=128,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH4)
    # 计算视差
    #disparity = stereo.compute(imgL, imgR)
    disparity = stereo.compute(img1_rectified, img2_rectified)
    # 归一化函数算法
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite('res_image/disaprity.png', disp * 4)
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16

    # 格式转变，BGRtoRGB
    frame1 = cv2.cvtColor(iml, cv2.COLOR_BGR2RGB)
    # 转变成Image格式
    frame1 = Image.fromarray(np.uint8(frame1))
    frame1_shape = np.array(np.shape(frame1)[0:2])
    # 调整图片大小、颜色通道，使其适应YOLO推理的格式
    # frame1 = resize_image(frame1,(640,480))
    frame1 = cv2.cvtColor(np.array(frame1), cv2.COLOR_RGB2BGR)
    
    middle_x = 200
    middle_y = 150
    distance = math.sqrt(threeD[middle_y][middle_x][0] ** 2 +
                             threeD[middle_y][middle_x][1] ** 2 + threeD[middle_y][middle_x][2] ** 2)
    distance = distance / 1000.0  # mm -> m
    time_end = time.time()
    print("dis=", distance)
    print("视差为：",disparity[middle_y][middle_x])
    print("所用时间为：", time_end-time_start)