import math
import os
import cv2
import numpy as np


#熊猫挂件
def panda_pendant(frame, lmarks):
    
    #图片整体尺寸
    _row,_col,_channel = frame.shape
    
    eye1 = cv2.imread("panda_shot/images/panda_eye1.png")
    eye2 = cv2.imread("panda_shot/images/panda_eye2.png")
    ear1 = cv2.imread("panda_shot/images/panda_ear1.png")
    ear2 = cv2.imread("panda_shot/images/panda_ear2.png")
    nose = cv2.imread("panda_shot/images/panda_nose.png")
    bamboo = cv2.imread("panda_shot/images/bamboo.png")
    
    #左眼
    #眼睛中心点(x1,y1),宽度l1   
    x1 = int((lmarks[0, 39] + lmarks[0, 36])/2)
    y1 = int((lmarks[1, 40] + lmarks[1, 41] + lmarks[1, 37] + lmarks[1, 38])/4)
    l1 = lmarks[0, 39] - lmarks[0, 36]        
    eye1_resize = cv2.resize(eye1, (int(2.5*l1), int(2.5*l1)), interpolation=cv2.INTER_CUBIC)
    #转换hsv
    hsv = cv2.cvtColor(eye1_resize, cv2.COLOR_BGR2HSV)
    #获取mask
    lower_white = np.array([0,0,0])
    upper_white = np.array([150,150,150])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    #mask优化，腐蚀膨胀
    erode = cv2.erode(mask,None,iterations=1)        
    dilate = cv2.dilate(erode,None,iterations=1)        
    #遍历替换
    center=[y1-int(1.25*l1), x1-int(1.5*l1)]#在新背景图片中左上角的位置
    rows,cols,channels = eye1_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255: #0代表黑色的点
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = eye1_resize[i,j]#此处替换颜色，为BGR通道

    #右眼
    x2 = int((lmarks[0, 45] + lmarks[0, 42])/2)
    y2 = int((lmarks[1, 43] + lmarks[1, 44] + lmarks[1, 46] + lmarks[1, 47])/4)
    l2 = lmarks[0, 45] - lmarks[0, 42]        
    eye2_resize = cv2.resize(eye2, (int(2.5*l2), int(2.5*l2)), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(eye2_resize, cv2.COLOR_BGR2HSV)        
    mask = cv2.inRange(hsv, lower_white, upper_white)
    erode = cv2.erode(mask,None,iterations=1)
    dilate = cv2.dilate(erode,None,iterations=1)
    center=[y2-int(1.25*l2), x2-int(1.0*l2)]#在新背景图片中左上角的位置
    rows,cols,channels = eye2_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = eye2_resize[i,j]#此处替换颜色，为BGR通道                    

    #左耳
    x3 = int((lmarks[0, 68] + lmarks[0, 69] + lmarks[0, 76])/3)
    y3 = int((lmarks[1, 68] + lmarks[1, 69] + lmarks[1, 76])/3)             
    ear1_resize = cv2.resize(ear1, (int(2.5*l1), int(2.5*l1)), interpolation=cv2.INTER_CUBIC)        
    hsv = cv2.cvtColor(ear1_resize, cv2.COLOR_BGR2HSV)       
    mask = cv2.inRange(hsv, lower_white, upper_white)        
    erode = cv2.erode(mask,None,iterations=1)
    dilate = cv2.dilate(erode,None,iterations=1)
    center=[y3-int(2.5*l1), x3-int(2.5*l1)]#在新背景图片中左上角的位置
    rows,cols,channels = ear1_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = ear1_resize[i,j]#此处替换颜色，为BGR通道                    

    #右耳
    x4 = int((lmarks[0, 72] + lmarks[0, 73] + lmarks[0, 79])/3)
    y4 = int((lmarks[1, 72] + lmarks[1, 73] + lmarks[1, 79])/3)              
    ear2_resize = cv2.resize(ear2, (int(2.5*l2), int(2.5*l2)), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(ear2_resize, cv2.COLOR_BGR2HSV)       
    mask = cv2.inRange(hsv, lower_white, upper_white)
    erode = cv2.erode(mask,None,iterations=1)
    dilate = cv2.dilate(erode,None,iterations=1)
    center=[y4-int(2.5*l2), x4]#在新背景图片中左上角的位置
    rows,cols,channels = ear2_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = ear2_resize[i,j]#此处替换颜色，为BGR通道

    #鼻子
    x5 = int((lmarks[0, 30] + lmarks[0, 32] + lmarks[0, 33] + lmarks[0, 34])/4)
    y5 = int((lmarks[1, 30] + lmarks[1, 32] + lmarks[1, 33] + lmarks[1, 34])/4)             
    nose_resize = cv2.resize(nose, (int(1.5*l2), int(1.5*l2)), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(nose_resize, cv2.COLOR_BGR2HSV)       
    mask = cv2.inRange(hsv, lower_white, upper_white)
    erode = cv2.erode(mask,None,iterations=1)
    dilate = cv2.dilate(erode,None,iterations=1)
    center=[y5-int(0.75*l2), x5-int(0.75*l2)]#在新背景图片中左上角的位置
    rows,cols,channels = nose_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = nose_resize[i,j]#此处替换颜色，为BGR通道

    #竹子
    x6 = int((lmarks[0, 62] + lmarks[0, 64] + lmarks[0, 66])/3)
    y6 = int((lmarks[1, 62] + lmarks[1, 64] + lmarks[1, 66])/3)             
    bamboo_resize = cv2.resize(bamboo, (int(3*l2), int(3*l2)), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(bamboo_resize, cv2.COLOR_BGR2HSV)
    lower_white = np.array([5,5,5])
    upper_white = np.array([250,250,250])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    erode = cv2.erode(mask,None,iterations=1)
    dilate = cv2.dilate(erode,None,iterations=1)
    center=[y6, x6]#在新背景图片中左上角的位置
    rows,cols,channels = bamboo_resize.shape
    for i in range(rows):
        for j in range(cols):
            if dilate[i,j]==255:
                if center[0]+i>0 and center[0]+i<_row and center[1]+j>0 and center[1]+j<_col:
                    frame[center[0]+i, center[1]+j] = bamboo_resize[i,j]#此处替换颜色，为BGR通道
                    
    return frame
        


