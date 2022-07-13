import cv2
import numpy as np
import random
import time
from panda_shot.panda_seg.seg_demo import seg_panda
#from panda_shot.portrait_matting_unet_flask_master.predict import get_mask_unet
from panda_shot.Background_Matting_master.test_segmentation_deeplab import get_mask
from panda_shot.PaddleSeg_release_25.contrib.PP_HumanSeg.bg_replace import get_person_mask_by_Paddle
from panda_shot.CartoonGAN import CartoonGAN
from panda_shot.Stylization import Stylization
from panda_shot.Mask_RCNN_master.samples.demo import maskrcnn
from panda_shot.python_tf_bodypix_develop.bodypix import bodypix

import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology, feature




'''
##### 通过连通性分析去除多余的人像（无法分离重叠的人像！！！）
#标记连通区域-4连通 
def LableConnectedRagion4(mask, labelmap, labelindex, quene):
    row, col  = mask.shape
    count = 0 #记录标记的点数
    while len(quene) != 0:
        count = count + 1
        (m, n) = quene[0]
        quene.remove(quene[0])
        if m > 0 and m < row and n + 1 > 0 and n + 1 < col:
            if mask[m, n+1] == 255 and labelmap[m, n+1] == 0:
                quene.append((m, n + 1))
                labelindex = labelindex + 1
                labelmap[m, n+1] = 255
        if m > 0 and m < row and n - 1 > 0 and n - 1 < col:
            if mask[m, n - 1] == 255 and labelmap[m, n - 1] == 0:
                quene.append((m, n - 1))
                labelindex = labelindex + 1
                labelmap[m, n - 1] = 255
        if m + 1 > 0 and m + 1 < row and n > 0 and n < col:
            if mask[m + 1, n] == 255 and labelmap[m + 1, n] == 0:
                quene.append((m + 1, n))
                labelindex = labelindex + 1
                labelmap[m + 1, n] = 255
        if m - 1 > 0 and m - 1 < row and n > 0 and n < col:
            if mask[m - 1, n] == 255 and labelmap[m - 1, n] == 0:
                quene.append((m - 1, n))
                labelindex = labelindex + 1
                labelmap[m - 1, n] = 255
    #print(count)
    return labelmap, count

# 标记连通区域-8连通
def LableConnectedRagion8(mask, labelmap, labelindex, quene):
    row, col  = mask.shape
    while len(quene) != 0:
        (m, n) = quene[0]
        quene.remove(quene[0])
        # print(m,n)
        # print(quene)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if m + i > 0 and m + i < row and n + j > 0 and n + j < col:
                    if mask[m + i, n + j] == 255 and labelmap[m + i, n + j] == 0:
                        quene.append((m + i, n + j))
                        labelindex = labelindex + 1
                        labelmap[m + i, n + j] = 255
    return labelmap


def remove_other_person(mask , startPoint):
    #print(mask.shape)
    startPoint = [int(startPoint[1]), int(startPoint[0])]#注意点的顺序
    labelmap = np.zeros(mask.shape)  # 标记矩阵
    labelindex = 0  # 标记记数
    quene = []  # 存储标记点位置信息

    quene.append(startPoint)
    labelindex = labelindex + 1
    #print(startPoint)
    labelmap[startPoint[0], startPoint[1]] = 255

    labelmap, count = LableConnectedRagion4(mask, labelmap, labelindex, quene)
    #labelmap = LableConnectedRagion8(mask, labelmap, labelindex, quene)

    return labelmap, count
'''





#*****执行人像分割和图像融合
def do_integration(frame, pose_kind, personSegModel, keypoints):
    _row, _col, _channel = frame.shape
    # 480x640
    person_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    person_brightness = person_gray.mean()  # 人物亮度值

    #frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_CUBIC)

    print("*** Seg person...")
    start = time.time()

    #这里获取的mask必须是灰度图
    #mask = get_mask_unet(frame)
    mask = get_mask(frame, personSegModel)
    #mask, colored_mask = bodypix(frame)

    #mask = get_person_mask_by_Paddle(frame, personSegModel) * 255

    end = time.time()
    print("*** Seg done.")
    print('分割用时: ', end - start)

    #检测人像分割结果
    #return mask

    #通过连通性分析去除多余的人像（无法分离重叠的人像！！！）
    #mask, count = remove_other_person(mask , keypoints[1])

    #通过 Mask R-CNN检测并去除多余的人像
    #mask = maskrcnn(frame, mask)

    #return mask
    #print("*** Remove other person done.")

    file = r'panda_shot/images/PartitionStylized/posekind.txt'
    with open(file, 'w') as f:
        print(pose_kind, file=f)
    cv2.imwrite('panda_shot/images/PartitionStylized/person_frame.jpg', frame)
    cv2.imwrite('panda_shot/images/PartitionStylized/person_mask.jpg', mask)

    person_pixels_num, person_pos, person_height, person_width = get_person_info(mask, _row, _col)
    # for i in range(4):
    #       cv2.circle(frame, (person_pos[i][1], person_pos[i][0]), 8, (255, 0, 0), 4)

    """
    if pose_kind == 0:
        while (1):
            env_num = random.randint(1, 55)
            env = cv2.imread("panda_shot/images/Environmental_Integration/random/" + str(env_num) + ".jpg")
            # 获取背景图片中熊猫的大小信息
            panda_pixels_num, panda_pos, panda_height, panda_width = get_panda_info(env)
            if panda_pixels_num > 100:
                print("ENV_NUM--------", env_num)
                break
        row, col, channel = env.shape
        # final_person = env
        # final_mask = np.zeros(env.shape,np.uint8)

        # 熊猫大小*1.2=人大小
        scale = (1.2 * panda_pixels_num) / person_pixels_num
        scale = round(scale, 2)
        print(panda_pixels_num, person_pixels_num, scale)
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        _row, _col, _channel = frame.shape
        bias_i = 0
        bias_j = 0

        if (panda_pos[2][1] - 0) > (col - panda_pos[3][1]):  # 熊猫的左边到图像的左边 > 熊猫的右边到图像的右边
            # if (panda_pos[0][0]-0) > (row-panda_pos[1][0]):#熊猫的上边到图像的上边 > 熊猫的下边到图像的下边
            bias_i = int(-(panda_pos[0][0] + panda_pos[1][0]) / 2 + (person_pos[0][0] + person_pos[1][0]) / 2)
            bias_j = int((person_pos[2][1] + person_pos[3][1]) / 2 - (_col / 2) + (panda_pos[2][1] / 2))
        else:
            # if (panda_pos[0][0]-0) > (row-panda_pos[1][0]):
            bias_i = int(-(panda_pos[0][0] + panda_pos[1][0]) / 2 + (person_pos[0][0] + person_pos[1][0]) / 2)
            bias_j = int((person_pos[2][1] + person_pos[3][1]) / 2 - (_col / 2) + ((panda_pos[3][1] + col) / 2))
        for i in range(_row):  # 每行
            for j in range(_col):  # 每列
                if mask[i, j] != 0:
                    if i + bias_i > 0 and i + bias_i < row and j + bias_j > 0 and j + bias_j < col:
                        env[i + bias_i, j + bias_j] = frame[i, j]
        '''
                        final_person[i+bias_i, j+bias_j] = frame[i, j]
                        final_mask[i+bias_i, j+bias_j] = np.array([255,255,255])
        #高斯模糊
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 1.5);
        #转换为灰度图
        final_mask = cv2.cvtColor(final_mask, cv2.COLOR_RGB2GRAY)
        print(final_mask)
        for i in range(row):
            for j in range(col):
                for k in range(3):
                    env[i, j][k] = int(env[i, j][k] * (1.0-final_mask[i, j]/255.0) + final_person[i, j][k] * (final_mask[i, j]/255.0))
        '''
        return env
    """

    if pose_kind == 1:
        print("--随机背景--")
        while (True):
            env_num = random.randint(1, 55)
            env = cv2.imread("panda_shot/images/Environmental_Integration/random/" + str(env_num) + ".jpg")
            # 获取背景图片中熊猫的大小信息
            panda_pixels_num, panda_pos, panda_height, panda_width = get_panda_info(env)
            if panda_pixels_num > 100:
                print("ENV_NUM--------", env_num)
                break
        row, col, channel = env.shape
        # 匹配人物和环境亮度
        env_gray = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
        env_brightness = env_gray.mean()  # 环境亮度均值
        frame = contrast_brightness_image(frame, 1, env_brightness - person_brightness + 20)  # 修改人像亮度值
        panda_pixels_num, panda_pos, panda_height, panda_width = get_panda_info(env)
        panda_location = 0  # 标记熊猫在背景图的靠左0还是靠右1
        # 宽度优先
        if panda_height / panda_width <= _row * 2 / _col:
            env_scale = _col / (2 * panda_width)  # 根据熊猫宽度计算缩放比例
            # 根据熊猫位置和大小裁剪熊猫图片，尽可能使熊猫刚好占据图片一半的位置
            y0 = int(panda_pos[1][0] - panda_width * 2 * _row / _col)
            y1 = panda_pos[1][0]
            # 熊猫右边空白大于左边，则裁剪时右边留白
            if col - panda_pos[3][1] > panda_pos[2][1]:
                x0 = panda_pos[3][1] - panda_width
                x1 = panda_pos[3][1] + panda_width
                panda_location = 0
            else:
                x0 = panda_pos[2][1] - panda_width
                x1 = panda_pos[2][1] + panda_width
                panda_location = 1
            if y0 < 0:
                y0 = 0
            if x0 < 0:
                x0 = 0
            if y1 > row:
                y1 = row
            if x1 > col:
                x1 = col
            env = env[y0:y1, x0:x1]
            env = cv2.resize(env, None, fx=env_scale, fy=env_scale, interpolation=cv2.INTER_CUBIC)
        else:
            env_scale = _row / panda_height
            y0 = panda_pos[0][0]
            y1 = panda_pos[1][0]
            # 熊猫右边空白大于左边
            if col - panda_pos[3][1] > panda_pos[2][1]:
                x0 = int((panda_pos[2][1] + panda_pos[3][1]) / 2 - panda_height * _col / (_row * 4))
                x1 = int((panda_pos[2][1] + panda_pos[3][1]) / 2 + panda_height * _col * 3 / (_row * 4))
                panda_location = 0
            else:
                x0 = int((panda_pos[2][1] + panda_pos[3][1]) / 2 - panda_height * _col * 3 / (_row * 4))
                x1 = int((panda_pos[2][1] + panda_pos[3][1]) / 2 + panda_height * _col / (_row * 4))
                panda_location = 1
            if y0 < 0:
                y0 = 0
            if x0 < 0:
                x0 = 0
            if y1 > row:
                y1 = row
            if x1 > col:
                x1 = col
            env = env[y0:y1, x0:x1]
            env = cv2.resize(env, None, fx=env_scale, fy=env_scale, interpolation=cv2.INTER_CUBIC)
        person_scale = 0
        if person_height / person_width <= _row * 2 / _col:  # 宽度优先
            person_scale = _col / (2 * person_width)
            frame = frame[0:_row, person_pos[2][1]:person_pos[2][1] + person_width]
            mask = mask[0:_row, person_pos[2][1]:person_pos[2][1] + person_width]
        else:
            person_scale = _row / person_height
            frame = frame[0:_row, int((person_pos[2][1] + person_pos[3][1] - _col / 2) / 2):int(
                (person_pos[2][1] + person_pos[3][1] + _col / 2) / 2)]
            mask = mask[0:_row, int((person_pos[2][1] + person_pos[3][1] - _col / 2) / 2):int(
                (person_pos[2][1] + person_pos[3][1] + _col / 2) / 2)]
        frame = cv2.resize(frame, None, fx=person_scale, fy=person_scale, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, None, fx=person_scale, fy=person_scale, interpolation=cv2.INTER_CUBIC)

        _row, _col, _channel = frame.shape
        row, col, channel = env.shape
        bias_i = row - _row
        bias_j = 0
        if panda_location == 0:
            bias_j = int(col / 2)
        cv2.imwrite('panda_shot/images/PartitionStylized/person_frame.jpg', frame)
        cv2.imwrite('panda_shot/images/PartitionStylized/env.jpg', env)
        cv2.imwrite('panda_shot/images/PartitionStylized/person_mask.jpg', mask)
        file = r'panda_shot/images/PartitionStylized/bias.txt'
        with open(file, 'w') as f:
            print(bias_j, file=f)
        for i in range(_row):  # 每行
            for j in range(_col):  # 每列
                if mask[i, j] != 0:
                    if i + bias_i > 0 and i + bias_i < row and j + bias_j > 0 and j + bias_j < col:
                        env[i + bias_i, j + bias_j] = frame[i, j]
        return env


    if pose_kind == 2:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_02.png")
        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_02_1.png")
        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_02_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    env[i, j] = frame[i, j]
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 3:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_03.png")
        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_03_1.png")
        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_03_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 150 < row and j + 150 < col:
                        env[i + 150, j + 150] = frame[i, j]
        for i in range(row):
            for j in range(col):
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 4:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_04.png")
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 150 < row and j - 120 > 0:
                        env[i + 150, j - 120] = frame[i, j]
        return env

    if pose_kind == 5:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_05.png")
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 615 < row and j + 1487 < col:
                        env[i + 615, j + 1487] = frame[i, j]
        return env

    if pose_kind == 6:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_06.png")
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < row and j + 548 < col:
                        env[i + 450, j + 548] = frame[i, j]
        return env

    if pose_kind == 7:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_07.png")
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if 450 - i >= 0 and j + 700 < col:
                        env[450 - i, j + 700] = frame[i, j]
        return env

    if pose_kind == 8:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_08.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 404 < row and j - 100 > 0:
                        env[i + 404, j - 100] = frame[i, j]
        return env

    if pose_kind == 9:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_09.png")
        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_09_1.png")
        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_09_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 50 < row and j + 300 < col:
                        env[i + 50, j + 300] = frame[i, j]
        for i in range(row):
            for j in range(col):
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 10:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_10.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 225 < row and j + 244 < col:
                        env[i + 225, j + 244] = frame[i, j]
        return env

    if pose_kind == 11:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_11.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if j + 0 <col:
                        env[i, j + 0] = frame[i, j]
        return env

    if pose_kind == 12:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_12.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 90 < row and j +70 < col:
                        env[i + 90, j +70] = frame[i, j]
        return env

    if pose_kind == 13:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_13.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < row and j +25<col:
                        env[i + 450, j +25] = frame[i, j]
        return env

    if pose_kind == 14:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_14.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 167 < row and j +62<col:
                        env[i + 167, j +62] = frame[i, j]
        return env

    if pose_kind == 15:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_15.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < col and 0< row-j+43 and row-j+43 < row:
                        env[row-j+43, i + 450] = frame[i, j]
        return env

    if pose_kind == 16:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_16.png")
        row, col, channel = env.shape

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 225 < row and j +383 <col:
                        env[i + 225, j +383] = frame[i, j]
        return env

    if pose_kind == 17:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_17.png")
        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_17_1.png")
        row, col, channel = env.shape
        env_gray = cv2.cvtColor(env_1, cv2.COLOR_BGR2GRAY)

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if j + 5 <col:
                        if env_gray[i, j + 5] != 0:
                            env[i, j + 5] = frame[i, j]
        return env



    '''
    if pose_kind == 10:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_08_A.png")
        row, col, channel = env.shape
        env_gray = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
        env_brightness = env_gray.mean()  # 环境亮度值
        frame = contrast_brightness_image(frame, 1, env_brightness - person_brightness + 20)
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 404 < row and j - 100 > 0:
                        env[i + 404, j - 100] = frame[i, j]
        return env
    if pose_kind == 11:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_08_L.png")        
        row, col, channel = env.shape
        env_gray = cv2.cvtColor(env, cv2.COLOR_BGR2GRAY)
        env_brightness = env_gray.mean()  # 环境亮度值
        frame = contrast_brightness_image(frame, 1, env_brightness - person_brightness + 20)
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 404 < row and j - 100 > 0:
                        env[i + 404, j - 100] = frame[i, j]
        return env
    '''


def stylize(img , kind):
    if kind == 10:
        return CartoonGAN(img)
    else:
        return Stylization(img, kind)

# *****图像融合后分区风格化
def Stylize_after_integration(person_stylize_kind, env_stylize_kind):
    pose_kind = 0
    file = r'panda_shot/images/PartitionStylized/posekind.txt'
    with open(file, 'r') as f:
        pose_kind = int(f.readline())
    frame = cv2.imread('panda_shot/images/PartitionStylized/person_frame.jpg')
    #_row, _col, _channel = frame.shape
    frame = stylize(frame, person_stylize_kind)
    # 风格化后图片的尺寸略有变化需要恢复
    # 普通Style后图片会变大
    # CartoonGAN后图片会变小
    #frame = cv2.resize(frame, (_col, _row), interpolation=cv2.INTER_CUBIC)
    #frame = frame[0:_row, 0:_col]
    _row, _col, _channel = frame.shape
    print(frame.shape)
    mask = cv2.imread('panda_shot/images/PartitionStylized/person_mask.jpg')
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_row, mask_col = mask.shape
    print(mask.shape)
    #避免风格化后由于frame尺寸的略微变化导致frame和mask的尺寸不同而发生错误
    if mask_row < _row:
        _row = mask_row
    if mask_col < _col:
        _col = mask_col

    if pose_kind == 1:
        env = cv2.imread('panda_shot/images/PartitionStylized/env.jpg')
        row, col, channel = env.shape
        #print(env.shape)
        #env = cv2.resize(env, (600, 450), interpolation=cv2.INTER_CUBIC)
        #print(env.shape)
        env = stylize(env, env_stylize_kind)
        #print(env.shape)
        #env = cv2.resize(env, (col, row), interpolation=cv2.INTER_CUBIC)
        #print(env.shape)

        bias_i = row - _row
        bias_j = 0
        file = r'panda_shot/images/PartitionStylized/bias.txt'
        with open(file, 'r') as f:
            bias_j = int(f.readline())
        for i in range(_row):  # 每行
            for j in range(_col):  # 每列
                if mask[i, j] != 0:
                    if i + bias_i > 0 and i + bias_i < row and j + bias_j > 0 and j + bias_j < col:
                        env[i + bias_i, j + bias_j] = frame[i, j]
        return env

    if pose_kind == 2:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_02.png")
        #row1, col1, channel1 = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row1, 0:col1]

        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_02_1.png")
        #row2, col2, channel2 = env_1.shape
        env_1 = stylize(env_1, env_stylize_kind)
        #env = env[0:row2, 0:col2]

        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_02_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    env[i, j] = frame[i, j]
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 3:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_03.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape

        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_03_1.png")
        #row1, col1, channel1 = env_1.shape
        env_1 = stylize(env_1, env_stylize_kind)
        #env = env[0:row1, 0:col1]

        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_03_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)

        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 150 < row and j + 150 < col:
                        env[i + 150, j + 150] = frame[i, j]
        for i in range(row):
            for j in range(col):
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 4:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_04.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 150 < row and j - 120 > 0:
                        env[i + 150, j - 120] = frame[i, j]
        return env

    if pose_kind == 5:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_05.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 615 < row and j + 1487 < col:
                        env[i + 615, j + 1487] = frame[i, j]
        return env

    if pose_kind == 6:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_06.png")
        row, col, channel = env.shape
        #print(env.shape)
        env = stylize(env, env_stylize_kind)
        #print(env.shape)
        #env = cv2.resize(env, (col, row), interpolation=cv2.INTER_CUBIC)  # 风格化后图片的尺寸略有变化需要恢复
        #env = env[0:row, 0:col]
        #print(env.shape)
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < row and j + 548 < col:
                        env[i + 450, j + 548] = frame[i, j]
        return env

    if pose_kind == 7:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_07.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if 450 - i >= 0 and j + 700 < col:
                        env[450 - i, j + 700] = frame[i, j]
        return env

    if pose_kind == 8:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_08.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 404 < row and j - 100 > 0:
                        env[i + 404, j - 100] = frame[i, j]
        return env

    if pose_kind == 9:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_09.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]

        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_09_1.png")
        #row1, col1, channel1 = env_1.shape
        env_1 = stylize(env_1, env_stylize_kind)
        #env = env[0:row1, 0:col1]

        env_2 = cv2.imread("panda_shot/images/Environmental_Integration/scene_09_2.png")
        gray = cv2.cvtColor(env_2, cv2.COLOR_BGR2GRAY)
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 50 < row and j + 300 < col:
                        env[i + 50, j + 300] = frame[i, j]
        for i in range(row):
            for j in range(col):
                if gray[i, j] == 0:
                    env[i, j] = env_1[i, j]
        return env

    if pose_kind == 10:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_10.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 225 < row and j + 244 < col:
                        env[i + 225, j + 244] = frame[i, j]
        return env

    if pose_kind == 11:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_11.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if j + 0 < col:
                        env[i, j + 0] = frame[i, j]
        return env

    if pose_kind == 12:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_12.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 90 < row and j + 70 < col:
                        env[i + 90, j + 70] = frame[i, j]
        return env

    if pose_kind == 13:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_13.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < row and j + 25 < col:
                        env[i + 450, j + 25] = frame[i, j]
        return env

    if pose_kind == 14:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_14.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 167 < row and j + 62 < col:
                        env[i + 167, j + 62] = frame[i, j]
        return env

    if pose_kind == 15:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_15.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 450 < col and 0 < row - j + 43 and row - j + 43 < row:
                        env[row - j + 43, i + 450] = frame[i, j]
        return env

    if pose_kind == 16:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_16.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if i + 225 < row and j + 383 < col:
                        env[i + 225, j + 383] = frame[i, j]
        return env

    if pose_kind == 17:
        env = cv2.imread("panda_shot/images/Environmental_Integration/scene_17.png")
        #row, col, channel = env.shape
        env = stylize(env, env_stylize_kind)
        #env = env[0:row, 0:col]

        env_1 = cv2.imread("panda_shot/images/Environmental_Integration/scene_17_1.png")
        env_gray = cv2.cvtColor(env_1, cv2.COLOR_BGR2GRAY)
        row, col, channel = env.shape
        for i in range(_row):
            for j in range(_col):
                if mask[i, j] != 0:
                    if j + 5 < col:
                        if env_gray[i, j + 5] != 0:
                            env[i, j + 5] = frame[i, j]
        return env


#*****调整图片对比度和亮度
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)  # addWeighted函数说明如下
    return dst

#*****获取背景图片中的熊猫位置和大小信息
def get_panda_info(image):
    row, col, channel = image.shape
    panda_mask = seg_panda(image)
    #image1 = cv2.cvtColor(np.asarray(panda_mask), cv2.COLOR_RGB2BGR)
    #cv2.imshow("Panda", image1)
    top_point = get_panda_vertex(1, panda_mask, row, col)
    bottom_point = get_panda_vertex(2, panda_mask, row, col)
    left_point = get_panda_vertex(3, panda_mask, row, col)
    right_point = get_panda_vertex(4, panda_mask, row, col)

    # 计算背景图中熊猫的分割图所占的像素数目
    panda_pixels_num = 0
    for i in range(320):
        for j in range(416):
            if panda_mask[i][j].any() != 0:
                panda_pixels_num = panda_pixels_num + 1
    panda_pixels_num = panda_pixels_num * (row * col) / (320 * 416)

    return panda_pixels_num, np.array([top_point, bottom_point, left_point, right_point]), bottom_point[0] - top_point[
        0], right_point[1] - left_point[1]
    # 序号0为Y坐标，序号1为X坐标

def get_panda_vertex(pos, panda_mask, row, col):
    if pos == 1:
        a = 0
        b = 320
        c = 0
        d = 416
        step1 = 1
        step2 = 1
    if pos == 2:
        a = 319
        b = -1
        c = 0
        d = 416
        step1 = -1
        step2 = 1
    if pos == 3:
        a = 0
        b = 416
        c = 0
        d = 320
        step1 = 1
        step2 = 1
    if pos == 4:
        a = 415
        b = -1
        c = 0
        d = 320
        step1 = -1
        step2 = 1
    for i in range(a, b, step1):
        for j in range(c, d, step2):
            if pos == 1 or pos == 2:
                if panda_mask[i][j].all() != 0:
                    return np.array([int(i * row / 320), int(j * col / 416)])
            if pos == 3 or pos == 4:
                if panda_mask[j][i].all() != 0:
                    return np.array([int(j * row / 320), int(i * col / 416)])


#*****获取照片中的人的大小信息
def get_person_info(person_mask, row, col):
    top_point = get_person_vertex(1, person_mask, row, col)
    bottom_point = get_person_vertex(2, person_mask, row, col)
    left_point = get_person_vertex(3, person_mask, row, col)
    right_point = get_person_vertex(4, person_mask, row, col)

    # 计算照片中人像分割所占的像素数目
    person_pixels_num = 0
    for i in range(row):
        for j in range(col):
            if person_mask[i][j] != 0:
                person_pixels_num = person_pixels_num + 1

    return person_pixels_num, np.array([top_point, bottom_point, left_point, right_point]), bottom_point[0] - top_point[
        0], right_point[1] - left_point[1]

def get_person_vertex(pos, person_mask, row, col):
    if pos == 1:
        a = 0
        b = row
        c = 0
        d = col
        step1 = 1
        step2 = 1
    if pos == 2:
        a = row - 1
        b = -1
        c = 0
        d = col
        step1 = -1
        step2 = 1
    if pos == 3:
        a = 0
        b = col
        c = 0
        d = row
        step1 = 1
        step2 = 1
    if pos == 4:
        a = col - 1
        b = -1
        c = 0
        d = row
        step1 = -1
        step2 = 1
    for i in range(a, b, step1):
        for j in range(c, d, step2):
            if pos == 1 or pos == 2:
                if person_mask[i][j] != 0:
                    return np.array([i, j])
            if pos == 3 or pos == 4:
                if person_mask[j][i] != 0:
                    return np.array([j, i])