# -*- coding:utf-8 -*-
__author__ = 'Chenyu and Liuyang'
import math
import os
import cv2
import numpy as np
import panda_shot.check_resources as check
import dlib
import panda_shot.facial_feature_detector as feature_detection
import panda_shot.PoseEstimation as PE

this_path = os.path.dirname(os.path.abspath(__file__))
check.check_dlib_landmark_weights()
predictor_path_68f = "panda_shot/dlib_models/shape_predictor_68_face_landmarks.dat"
predictor_for_pose = dlib.shape_predictor(predictor_path_68f)
predictor_path_81f = 'panda_shot/dlib_models/shape_predictor_81_face_landmarks.dat'
predictor_for_lmarks = dlib.shape_predictor(predictor_path_81f)
detector_for_face = dlib.get_frontal_face_detector()

def demo(cap,out):
    Pose_Angle = []
    Pose_Para=[]
    lmarks_cov=[]
    # 视频帧截取
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    dets = detector_for_face(frame, 0)
    # 检测特征点
    lmarks = feature_detection.get_landmarks(frame, detector_for_face, predictor_for_pose)
    if len(lmarks):
        Pose_Para = PE.poseEstimation(frame, lmarks)
    else:
        print('No face detected!')
    Pose_Angle = np.array(Pose_Para) * 180 / math.pi
    for k, d in enumerate(dets):
        shape = predictor_for_lmarks(frame, d)
        # 返回一个2*81的landmarks数组元素，第一维x坐标，第二位y坐标
        lmarks_cov_x = np.matrix([[p.x] for p in shape.parts()]).reshape(1, 81)
        lmarks_cov_y = np.matrix([[p.y] for p in shape.parts()]).reshape(1, 81)
        lmarks_cov = np.vstack((lmarks_cov_x, lmarks_cov_y))
        # opencv画点
        for num in range(shape.num_parts):
            cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0, 255, 0), -1)
    cv2.imshow('frame', frame)
    out.write(frame)
    return  Pose_Angle,lmarks_cov

