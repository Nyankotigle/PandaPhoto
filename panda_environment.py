import cv2
import math
import numpy as np
# from panda_shot.openpose170.build.examples.tutorial_api_python.get_body_keypoint import get_body_keypoint
from panda_shot.lightweight_openpose.keypoints_from_lightweight_openpose import get_lwop_body_keypoint
from panda_shot.panda_environmental_integration_lwop import do_integration
from panda_shot.panda_environment_pose_recognition_lwop import match_pose
from panda_shot.panda_environment_pose_recognition_lwop import match_pose_knn
from panda_shot.panda_environmental_global import WriteMatchSuccess

#********************************
#*********人像分割环境融合**********
#********************************
def panda_environment(frame, photoClass ,personSegModel, envMode1_video):

    _row, _col, _channel = frame.shape
    #获取人体关节特征点

    keypoints = get_lwop_body_keypoint(frame)
    #画面中没有识别到人体
    if keypoints is None:
        print("No person!!!")
        WriteMatchSuccess(0)  #停止查询匹配是否成功
        return None, -1, 0

    '''
    #*********************
    #*****动作点数据收集*****
    print("写入数据...")
    keypointData = np.floor(np.delete(keypoints[0], -1, axis=1).reshape(50,))
    file = r'panda_shot/Keypoints_DATA.txt'
    with open(file, 'a+') as f:
        for data in keypointData:
            print(int(data), file=f)
    return None, 5, 0
    #*********************
    '''


    # 从多个识别到的人中选择特征点平均间距最大的人
    keypoints = select_main_person(keypoints)

    # 将lightweight_openpose的18个关键点扩充到19个
    keypointData = np.zeros((19, 2))
    for i in range(8):
        keypointData[i] = keypoints[i]
    keypointData[8] = (keypoints[8] + keypoints[11]) / 2
    for i in range(9, 19):
        keypointData[i] = keypoints[i - 1]
    keypoints = keypointData

    print("start match pose !!!!!")
    # 匹配预设的动作
    pose_kind, perfectMatch = match_pose(keypoints)
    # pose_kind, perfectMatch = match_pose_knn(keypoints)

    if perfectMatch == 0:
        if photoClass == 4:  # 未匹配到动作且是定时拍照时，直接用模式1，即随机背景
            pose_kind = 1
        else:  # 未匹配到动作且是动作识别时，frame返回为空，继续识别下一帧
            print("NoMatch!!")
            return None, pose_kind, 0
    # 匹配成功
    print("**pose_kind:", pose_kind)

    WriteMatchSuccess(pose_kind)  # 通知匹配成功

    #视频模式只需要返回匹配到的动作
    if envMode1_video==1:
        return None, pose_kind, 2

    '''
    # 人像预分割，去除多余人像的干扰
    if photoClass == 4:
        frame = pre_people_seg(frame, keypoints)
    '''

    # 根据匹配到的动作执行相应的背景融合
    frame = do_integration(frame, pose_kind, personSegModel, keypoints)
    cv2.imwrite('panda_shot/images/PartitionStylized/intergrated.jpg', frame)

    print("integration done !!!!!")

    return frame, pose_kind, 1



# *****从多个识别到的人中选择特征点平均间距最大的人
def select_main_person(keypoints):
    max_mean_dist = 0  # 最大平均间距
    max_people = 0  # 特征点平均间距最大的人的序号
    mean_dist = 0  # 计算中的平均间距
    w1, w2, w3 = keypoints.shape  # w1识别到的人的数目
    if w1 > 1:
        # 为每个人计算特征点的平均间距
        for i in range(w1):
            # 过滤掉坐标为-1，即未识别到的特征点
            points = []
            for j in range(18):
                if keypoints[i][j][0] != -1 and keypoints[i][j][1] != -1:
                    points.append([keypoints[i][j][0], keypoints[i][j][1]])
            mean_dist = get_mean_dist(points)
            # print(mean_dist)
            # 记录最大平均间距和对应的人的序号
            if mean_dist > max_mean_dist:
                max_mean_dist = mean_dist
                max_people = i
        print("**max_people:", max_people, "**max_mean_dist:", max_mean_dist)
        keypoints = keypoints[max_people]
    else:
        keypoints = keypoints[0]
    return keypoints



#*****人像预分割，去除多余人像的干扰
def pre_people_seg(frame, keypoints):
    _row, _col, _channel = frame.shape
    # 对于图片中的每个点，判断它与各个特征点之间的位置关系，粗略分割主要人像区域，平均开销35s(省略这一过程的话平均开销13s)
    dis_1 = getScore([keypoints[1][0], keypoints[1][1]], [keypoints[0][0], keypoints[0][1]])
    dis_2 = getScore([keypoints[2][0], keypoints[2][1]], [keypoints[5][0], keypoints[5][1]])
    _keypoints_0 = [keypoints[1][0], keypoints[1][1] - 2.3 * (keypoints[1][1] - keypoints[0][1])]
    _keypoints_4 = [keypoints[4][0] * 1.9 - keypoints[3][0] * 0.9, keypoints[4][1] * 1.9 - keypoints[3][1] * 0.9]
    _keypoints_7 = [keypoints[7][0] * 1.9 - keypoints[6][0] * 0.9, keypoints[7][1] * 1.9 - keypoints[6][1] * 0.9]
    scale = 10.0#将图片缩小获得掩码图再放大以减少时间复杂度
    mask_1 = np.zeros((int(_row/scale),int(_col/scale),3), dtype=np.uint8)
    for i in range(int(_row/scale)):  # 每行
        for j in range(int(_col/scale)):  # 每列
            if ((p_to_l_distance([j*scale, i*scale], _keypoints_0, keypoints[1]) < dis_1 and get_angle(_keypoints_0,[j*scale, i*scale],_keypoints_0,keypoints[1]) <= 90 and get_angle(keypoints[1], [j*scale, i*scale], keypoints[1], _keypoints_0) <= 90) #头\
                    or (dis_2 < dis_1 and j*scale > keypoints[1][0] - dis_1 * 1.2 and j*scale < keypoints[1][0] + dis_1 * 1.2 and i*scale > keypoints[1][1] - dis_1 * 0.5) #身体\
                    or (dis_2 >= dis_1 and j * scale > keypoints[1][0] - dis_2 * 0.8 and j * scale < keypoints[1][0] + dis_2 * 0.8 and i * scale > keypoints[1][1] - dis_2 * 0.3)  # 身体\
                    or (keypoints[2]!=[0.0,0.0] and keypoints[3]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[2], keypoints[3]) < dis_1 * 0.5 and get_angle(keypoints[2],[j*scale, i*scale],keypoints[2],keypoints[3]) <= 90 and get_angle(keypoints[3], [j*scale, i*scale], keypoints[3], keypoints[2]) <= 90) #胳膊\
                    or (keypoints[3]!=[0.0,0.0] and keypoints[4]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[3], _keypoints_4) < dis_1 * 0.5 and get_angle(keypoints[3],[j*scale, i*scale],keypoints[3],_keypoints_4) <= 90 and get_angle(_keypoints_4, [j*scale, i*scale], _keypoints_4, keypoints[3]) <= 90) #胳膊\
                    or (keypoints[5]!=[0.0,0.0] and keypoints[6]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[5], keypoints[6]) < dis_1 * 0.5 and get_angle(keypoints[5],[j*scale, i*scale],keypoints[5],keypoints[6]) <= 90 and get_angle(keypoints[6], [j*scale, i*scale], keypoints[6], keypoints[5]) <= 90) #胳膊\
                    or (keypoints[6]!=[0.0,0.0] and keypoints[7]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[6], _keypoints_7) < dis_1 * 0.5 and get_angle(keypoints[6],[j*scale, i*scale],keypoints[6],_keypoints_7) <= 90 and get_angle(_keypoints_7, [j*scale, i*scale], _keypoints_7, keypoints[6]) <= 90) #胳膊\
                    or (keypoints[3]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[3]) < dis_1 * 0.5) #关节\
                    or (keypoints[6]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[6]) < dis_1 * 0.5) #关节\
                    or (keypoints[2]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[2]) < dis_1 * 0.5) #关节\
                    or (keypoints[5]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[5]) < dis_1 * 0.5) #关节\
                    or (keypoints[4]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[4]) < dis_1 * 1.2 and get_angle(keypoints[4], [j*scale, i*scale], keypoints[3],keypoints[4]) <= 90) #手\
                    or (keypoints[7]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[7]) < dis_1 * 1.2 and get_angle(keypoints[7], [j*scale, i*scale], keypoints[6],keypoints[7]) <= 90) #手\
                    ):
                mask_1[i, j] = np.array([255,255,255])
    mask_2 = cv2.resize(mask_1, (_col,_row), interpolation=cv2.INTER_CUBIC)
    frame = cv2.bitwise_and(frame,mask_2) #利用原图和掩码图之间的位运算抠图
    return frame


#*****计算两点之间的欧氏距离
def getScore(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
#*****计算一组坐标点中所有点之间的平均间距
def get_mean_dist(points):
    dist = []
    for i, (x0, y0) in enumerate(points):
        for j, (x1, y1) in enumerate(points):
            if not i == j:
                dist.append(getScore([x0, y0], [x1, y1]))
    return  np.array(dist).mean()
#*****点到直线的距离
def p_to_l_distance(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance
#*****计算向量AB和CD之间的夹角
def get_angle(A,B,C,D):
    dx1 = B[0] - A[0]
    dy1 = B[1] - A[1]
    dx2 = D[0] - C[0]
    dy2 = D[1] - C[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle



### 原代码备份
"""
import cv2
import math
import numpy as np
from panda_shot.openpose170.build.examples.tutorial_api_python.get_body_keypoint import get_body_keypoint
from panda_shot.panda_environmental_integration import do_integration
from panda_shot.panda_environment_pose_recognition import match_pose
from panda_shot.panda_environmental_global import WriteMatchSuccess

#********************************
#*********人像分割环境融合**********
#********************************
def panda_environment(frame, photoClass ,personSegModel, envMode1_video):

    _row,_col,_channel = frame.shape
    #获取人体关节特征点

    keypoints = get_body_keypoint(frame)
    #画面中没有识别到人体
    if keypoints is None:
        print("No person!!!")
        WriteMatchSuccess(0)  #停止查询匹配是否成功
        return None, -1, 0

    '''
    #*********************
    #*****动作点数据收集*****
    print("写入数据...")
    keypointData = np.floor(np.delete(keypoints[0], -1, axis=1).reshape(50,))
    file = r'panda_shot/Keypoints_DATA.txt'
    with open(file, 'a+') as f:
        for data in keypointData:
            print(int(data), file=f)
    return None, 5, 0
    #*********************
    '''


    # 从多个识别到的人中选择特征点平均间距最大的人
    keypoints = select_main_person(keypoints)

    # 匹配预设的动作
    pose_kind , perfectMatch = match_pose(keypoints)
    if perfectMatch == 0:
        if photoClass == 4:  # 未匹配到动作且是定时拍照时，直接用模式1，即随机背景
            pose_kind = 1
        else:  # 未匹配到动作且是动作识别时，frame返回为空，继续识别下一帧
            print("NoMatch!!")
            return None, pose_kind, 0
    # 匹配成功
    print("**pose_kind:", pose_kind)

    WriteMatchSuccess(pose_kind)  # 通知匹配成功

    #视频模式只需要返回匹配到的动作
    if envMode1_video==1:
        return None, pose_kind, 2

    '''
    # 人像预分割，去除多余人像的干扰
    if photoClass == 4:
        frame = pre_people_seg(frame, keypoints)
    '''

    # 根据匹配到的动作执行相应的背景融合
    frame = do_integration(frame, pose_kind, personSegModel, keypoints)
    cv2.imwrite('panda_shot/images/PartitionStylized/intergrated.jpg', frame)

    return frame, pose_kind, 1


#*****从多个识别到的人中选择特征点平均间距最大的人
def select_main_person(keypoints):
    max_mean_dist = 0   #最大平均间距
    max_people = 0      #特征点平均间距最大的人的序号
    mean_dist = 0       #计算中的平均间距
    w1,w2,w3 = keypoints.shape  #w1识别到的人的数目
    if w1>1:
        #为每个人计算特征点的平均间距
        for i in range(w1):
            #过滤掉坐标为零，即未识别到的特征点
            points = []
            for j in range(25):
                if keypoints[i][j][0]!=0.0 and keypoints[i][j][1]!=0.0:
                    points.append([keypoints[i][j][0],keypoints[i][j][1]])
            mean_dist = get_mean_dist(points)
            #print(mean_dist)
            #记录最大平均间距和对应的人的序号
            if mean_dist > max_mean_dist:
                max_mean_dist = mean_dist
                max_people = i
        print("**max_people:",max_people,"**max_mean_dist:", max_mean_dist)
        keypoints = keypoints[max_people]
    else:
        keypoints =  keypoints[0]

    '''
    #将关键点坐标格式化输出到文件
    file = r'panda_shot/StandardKeypoints.txt'
    with open(file, 'a+') as f:
        print('[', end = '', file=f)
        for i in range(25):
            print('[', ",".join(str(round(j, 1)) for j in keypoints[i]), ']', end = '', file = f)
            if i != 24:
                print(',', file=f)
            else:
                print('],', file=f)
        print('\n', file=f)
    '''
    #舍弃keypoints中的第三列
    _keypoints = []
    for points in keypoints:
        _keypoints.append([points[0], points[1]])
    keypoints = _keypoints
    return keypoints


#*****人像预分割，去除多余人像的干扰
def pre_people_seg(frame, keypoints):
    _row, _col, _channel = frame.shape
    # 对于图片中的每个点，判断它与各个特征点之间的位置关系，粗略分割主要人像区域，平均开销35s(省略这一过程的话平均开销13s)
    dis_1 = getScore([keypoints[1][0], keypoints[1][1]], [keypoints[0][0], keypoints[0][1]])
    dis_2 = getScore([keypoints[2][0], keypoints[2][1]], [keypoints[5][0], keypoints[5][1]])
    _keypoints_0 = [keypoints[1][0], keypoints[1][1] - 2.3 * (keypoints[1][1] - keypoints[0][1])]
    _keypoints_4 = [keypoints[4][0] * 1.9 - keypoints[3][0] * 0.9, keypoints[4][1] * 1.9 - keypoints[3][1] * 0.9]
    _keypoints_7 = [keypoints[7][0] * 1.9 - keypoints[6][0] * 0.9, keypoints[7][1] * 1.9 - keypoints[6][1] * 0.9]
    scale = 10.0#将图片缩小获得掩码图再放大以减少时间复杂度
    mask_1 = np.zeros((int(_row/scale),int(_col/scale),3), dtype=np.uint8)
    for i in range(int(_row/scale)):  # 每行
        for j in range(int(_col/scale)):  # 每列
            if ((p_to_l_distance([j*scale, i*scale], _keypoints_0, keypoints[1]) < dis_1 and get_angle(_keypoints_0,[j*scale, i*scale],_keypoints_0,keypoints[1]) <= 90 and get_angle(keypoints[1], [j*scale, i*scale], keypoints[1], _keypoints_0) <= 90) #头\
                    or (dis_2 < dis_1 and j*scale > keypoints[1][0] - dis_1 * 1.2 and j*scale < keypoints[1][0] + dis_1 * 1.2 and i*scale > keypoints[1][1] - dis_1 * 0.5) #身体\
                    or (dis_2 >= dis_1 and j * scale > keypoints[1][0] - dis_2 * 0.8 and j * scale < keypoints[1][0] + dis_2 * 0.8 and i * scale > keypoints[1][1] - dis_2 * 0.3)  # 身体\
                    or (keypoints[2]!=[0.0,0.0] and keypoints[3]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[2], keypoints[3]) < dis_1 * 0.5 and get_angle(keypoints[2],[j*scale, i*scale],keypoints[2],keypoints[3]) <= 90 and get_angle(keypoints[3], [j*scale, i*scale], keypoints[3], keypoints[2]) <= 90) #胳膊\
                    or (keypoints[3]!=[0.0,0.0] and keypoints[4]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[3], _keypoints_4) < dis_1 * 0.5 and get_angle(keypoints[3],[j*scale, i*scale],keypoints[3],_keypoints_4) <= 90 and get_angle(_keypoints_4, [j*scale, i*scale], _keypoints_4, keypoints[3]) <= 90) #胳膊\
                    or (keypoints[5]!=[0.0,0.0] and keypoints[6]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[5], keypoints[6]) < dis_1 * 0.5 and get_angle(keypoints[5],[j*scale, i*scale],keypoints[5],keypoints[6]) <= 90 and get_angle(keypoints[6], [j*scale, i*scale], keypoints[6], keypoints[5]) <= 90) #胳膊\
                    or (keypoints[6]!=[0.0,0.0] and keypoints[7]!=[0.0,0.0] and p_to_l_distance([j*scale, i*scale], keypoints[6], _keypoints_7) < dis_1 * 0.5 and get_angle(keypoints[6],[j*scale, i*scale],keypoints[6],_keypoints_7) <= 90 and get_angle(_keypoints_7, [j*scale, i*scale], _keypoints_7, keypoints[6]) <= 90) #胳膊\
                    or (keypoints[3]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[3]) < dis_1 * 0.5) #关节\
                    or (keypoints[6]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[6]) < dis_1 * 0.5) #关节\
                    or (keypoints[2]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[2]) < dis_1 * 0.5) #关节\
                    or (keypoints[5]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[5]) < dis_1 * 0.5) #关节\
                    or (keypoints[4]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[4]) < dis_1 * 1.2 and get_angle(keypoints[4], [j*scale, i*scale], keypoints[3],keypoints[4]) <= 90) #手\
                    or (keypoints[7]!=[0.0,0.0] and getScore([j*scale, i*scale], keypoints[7]) < dis_1 * 1.2 and get_angle(keypoints[7], [j*scale, i*scale], keypoints[6],keypoints[7]) <= 90) #手\
                    ):
                mask_1[i, j] = np.array([255,255,255])
    mask_2 = cv2.resize(mask_1, (_col,_row), interpolation=cv2.INTER_CUBIC)
    frame = cv2.bitwise_and(frame,mask_2) #利用原图和掩码图之间的位运算抠图
    return frame


#*****计算两点之间的欧氏距离
def getScore(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))
#*****计算一组坐标点中所有点之间的平均间距
def get_mean_dist(points):
    dist = []
    for i, (x0, y0) in enumerate(points):
        for j, (x1, y1) in enumerate(points):
            if not i == j:
                dist.append(getScore([x0, y0], [x1, y1]))
    return  np.array(dist).mean()
#*****点到直线的距离
def p_to_l_distance(point, line_point1, line_point2):
    #对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        point_array = np.array(point )
        point1_array = np.array(line_point1)
        return np.linalg.norm(point_array -point1_array )
    #计算直线的三个参数
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
        (line_point2[0] - line_point1[0]) * line_point1[1]
    #根据点到直线的距离公式计算距离
    distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
    return distance
#*****计算向量AB和CD之间的夹角
def get_angle(A,B,C,D):
    dx1 = B[0] - A[0]
    dy1 = B[1] - A[1]
    dx2 = D[0] - C[0]
    dy2 = D[1] - C[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    # print(angle2)
    if angle1*angle2 >= 0:
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle
"""


