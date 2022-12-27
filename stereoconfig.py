import numpy as np
 
 
# 双目相机参数
class stereoCamera(object):
    def __init__(self):
        # 左相机内参
        self.cam_matrix_left = np.array([[738.0143, 0.4567, 608.2346],
                                         [0., 738.3358, 338.4390],
                                         [0., 0., 1.0000]])
        # 右相机内参
        self.cam_matrix_right = np.array([[730.1282, 1.1600, 615.2917],
                                          [0., 731.635, 328.7212],
                                          [0., 0., 1.0000]])
 
        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = np.array([[0.0608, 0.1244, 0.0017, 0.0005, -0.4115]])
        self.distortion_r = np.array([[0.1146, -0.1297, 4.1067e-04,  -0.0017, -0.0496]])
 
        # 旋转矩阵
        self.R = np.array([[1.0000, -0.0007, -0.0050],
                           [0.0007, 1.0000, 0.0009],
                           [0.0050, -0.0009, 1.0000]])
 
        # 平移矩阵
        self.T = np.array([[-59.9402], [-0.1261], [-0.8328]])
 
        # 主点列坐标的差
        self.doffs = 59.9402
 
        # 指示上述内外参是否为经过立体校正后的结果
        self.isRectified = False
 
    def setMiddleBurryParams(self):
        self.cam_matrix_left = np.array([[3997.684, 0, 225.0],
                                                            [0., 3997.684, 187.5],
                                                            [0., 0., 1.]])
        self.cam_matrix_right =  np.array([[3997.684, 0, 225.0],
                                                                [0., 3997.684, 187.5],
                                                                [0., 0., 1.]])
        self.distortion_l = np.zeros(shape=(5, 1), dtype=np.float64)
        self.distortion_r = np.zeros(shape=(5, 1), dtype=np.float64)
        self.R = np.identity(3, dtype= np.float64)
        self.T = np.array([[-193.001], [0.0], [0.0]])
        self.doffs = 131.111
        self.isRectified = True
 
 