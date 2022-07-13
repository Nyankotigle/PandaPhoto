import math
import os
import cv2
import numpy as np


#熊猫头表情
def panda_emoticons(frame, lmarks):

    #图片整体尺寸
    _row,_col,_channel = frame.shape

    # 图片整体尺寸
    _row, _col, _channel = frame.shape

    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 素描风格
    inv = 255 - gray
    ksize = 15
    sigma = 50
    blur = cv2.GaussianBlur(inv, ksize=(ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    frame = cv2.divide(gray, 255 - blur, scale=255)
    '''
    #伽马变换
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            frame[i,j] = 3*math.pow(frame[i,j], 2.5)
    cv2.normalize(frame,frame,0,255,cv2.NORM_MINMAX)
    frame = cv2.convertScaleAbs(frame)
    '''

    # 脸部mask
    # points = np.array([[[lmarks[0, 77],lmarks[1, 77]], [lmarks[0, 0],lmarks[1, 0]], [lmarks[0, 1],lmarks[1, 1]], [lmarks[0, 2],lmarks[1, 2]], [lmarks[0, 3],lmarks[1, 3]], [lmarks[0, 4],lmarks[1, 4]], [lmarks[0, 5],lmarks[1, 5]], [lmarks[0, 6],lmarks[1, 6]], [lmarks[0, 7],lmarks[1, 7]], [lmarks[0, 8],lmarks[1, 8]], [lmarks[0, 9],lmarks[1, 9]], [lmarks[0, 10],lmarks[1, 10]], [lmarks[0, 12],lmarks[1, 12]], [lmarks[0, 13],lmarks[1, 13]], [lmarks[0, 14],lmarks[1, 14]], [lmarks[0, 15],lmarks[1, 15]], [lmarks[0, 16],lmarks[1, 16]], [lmarks[0, 78],lmarks[1, 78]], [lmarks[0, 74],lmarks[1, 74]], [lmarks[0, 79],lmarks[1, 79]], [lmarks[0, 73],lmarks[1, 73]], [lmarks[0, 72],lmarks[1, 72]], [lmarks[0, 80],lmarks[1, 80]], [lmarks[0, 71],lmarks[1, 71]], [lmarks[0, 70],lmarks[1, 70]], [lmarks[0, 69],lmarks[1, 69]], [lmarks[0, 68],lmarks[1, 68]], [lmarks[0, 76],lmarks[1, 76]], [lmarks[0, 75],lmarks[1, 75]], [lmarks[0, 77],lmarks[1, 77]]]], dtype=np.int32)
    # points = np.array([[[lmarks[0, 0],lmarks[1, 0]], [lmarks[0, 77],lmarks[1, 77]], [lmarks[0, 17],lmarks[1, 17]], [lmarks[0, 18],lmarks[1, 18]], [lmarks[0, 19],lmarks[1, 19]], [lmarks[0, 20],lmarks[1, 20]], [lmarks[0, 21],lmarks[1, 21]], [lmarks[0, 22],lmarks[1, 22]], [lmarks[0, 23],lmarks[1, 23]], [lmarks[0, 24],lmarks[1, 24]], [lmarks[0, 25],lmarks[1, 25]], [lmarks[0, 26],lmarks[1, 26]], [lmarks[0, 78],lmarks[1, 78]], [lmarks[0, 16],lmarks[1, 16]], [lmarks[0, 10],lmarks[1, 10]], [lmarks[0, 9],lmarks[1, 9]], [lmarks[0, 8],lmarks[1, 8]], [lmarks[0, 7],lmarks[1, 7]], [lmarks[0, 6],lmarks[1, 6]], [lmarks[0, 0],lmarks[1, 0]]]], dtype=np.int32)
    points = np.array([[[lmarks[0, 0], lmarks[1, 0]], [lmarks[0, 77], lmarks[1, 77]], [lmarks[0, 17], lmarks[1, 17]],
                        [lmarks[0, 18], lmarks[1, 18]], [lmarks[0, 19], lmarks[1, 19]], [lmarks[0, 20], lmarks[1, 20]],
                        [lmarks[0, 21], lmarks[1, 21]], [lmarks[0, 22], lmarks[1, 22]], [lmarks[0, 23], lmarks[1, 23]],
                        [lmarks[0, 24], lmarks[1, 24]], [lmarks[0, 25], lmarks[1, 25]], [lmarks[0, 26], lmarks[1, 26]],
                        [lmarks[0, 78], lmarks[1, 78]], [lmarks[0, 16], lmarks[1, 16]], [lmarks[0, 54], lmarks[1, 54]],
                        [lmarks[0, 55], lmarks[1, 55]], [lmarks[0, 56], lmarks[1, 56]], [lmarks[0, 57], lmarks[1, 57]],
                        [lmarks[0, 58], lmarks[1, 58]], [lmarks[0, 59], lmarks[1, 59]], [lmarks[0, 48], lmarks[1, 48]],
                        [lmarks[0, 0], lmarks[1, 0]]]], dtype=np.int32)

    # 构造全零矩阵
    zeros = np.zeros((frame.shape), dtype=np.uint8)
    # 依据脸部mask点阵构造并填充多边形
    face_mask = cv2.fillPoly(zeros, points, color=(255, 255, 255))

    # 熊猫头
    head = cv2.imread("panda_shot/images/panda_head.png")
    # 熊猫头位置
    x7 = int((lmarks[0, 8] + lmarks[0, 68] + lmarks[0, 73]) / 3)
    y7 = int((lmarks[1, 8] + lmarks[1, 68] + lmarks[1, 73]) / 3)
    # 缩放比例
    l7 = int(1.8 * (lmarks[0, 16] - lmarks[0, 0]))
    head_resize = cv2.resize(head, (l7, l7), interpolation=cv2.INTER_CUBIC)
    center = [y7 - int(l7 / 2.4), x7 - int(l7 / 2)]  # 在新背景图片中左上角的位置

    # 非脸部填充为白色
    for i in range(_row):
        for j in range(_col):
            if face_mask[i, j].any() == 0:
                frame[i, j] = 255

    # 上熊猫头
    rows, cols, channels = head_resize.shape
    for i in range(rows):
        for j in range(cols):
            if head_resize[i, j].all() == 0:
                if center[0] + i > 0 and center[0] + i < _row and center[1] + j > 0 and center[1] + j < _col:
                    frame[center[0] + i, center[1] + j] = 0

    # 颜色加深
    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i, j] < 250:
                frame[i, j] = 0

    return frame
