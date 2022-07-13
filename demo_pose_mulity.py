# -*- coding:utf-8 -*-
__author__ = 'Chenyu and Liuyang'
import math
import os
import time
import sys
import cv2
import glob

import numpy as np
from skimage import io

import check_resources as check
import dlib
import facial_feature_detector as feature_detection
import PoseEstimation as PE
detector = dlib.get_frontal_face_detector()
this_path = os.path.dirname(os.path.abspath(__file__))
check.check_dlib_landmark_weights()  # 检测dlib参数是否下载，没下载的话下载
image_name = 'dr.jpg'
image_path = this_path + '/input/' + image_name


predictor_path = this_path + "/dlib_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
predictor_path2 = 'E:/Real-TimeFaceDetection/dlib_models/shape_predictor_81_face_landmarks.dat'
predictor2 = dlib.shape_predictor(predictor_path2)

def demo():
    NUM_FACE = 0
    img = io.imread(image_path)
    # 检测特征点
    dets = detector(img, 0)
    SET_ARR = True
    SET_ARR_2=True
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        lmarks_pose = feature_detection.get_lmarks(img,shape,dets)  #角度检测使用landmarks
        lmarks_cov_x = np.matrix([[p.x] for p in shape.parts()]).reshape(1,68)
        lmarks_cov_y = np.matrix([[p.y] for p in shape.parts()]).reshape(1,68)
        lmarks_cov=np.vstack((lmarks_cov_x,lmarks_cov_y))   #返回81个点坐标使用landmarks
        if SET_ARR:
            SET_ARR=False
            lmarks_all=lmarks_cov
        elif SET_ARR_2:
            lmarks_all=np.array([lmarks_all,lmarks_cov])
            SET_ARR_2 = False
        else:
            lmarks_all=np.concatenate((lmarks_all,lmarks_cov))
        NUM_FACE =1+ NUM_FACE
        if len(lmarks_cov):
            Pose_Para = PE.poseEstimation(img, lmarks_pose)
            print(np.array(Pose_Para) * 180 / math.pi)

    print('Number of faces detected: %d'% NUM_FACE)


    # if len(lmarks):
    #     Pose_Para = PE.poseEstimation(img, lmarks)
    # else:
    #     print('NO face detected!')
    #     return 0
    # print(np.array(Pose_Para)*180/math.pi)
if __name__ == "__main__":
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    c = np.array([a,b])
    demo()