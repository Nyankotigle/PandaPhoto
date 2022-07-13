import cv2
import sys
from django.shortcuts import render
from django.http import HttpResponse
import os
import math
import numpy as np
import panda_shot.check_resources as check
import dlib
import panda_shot.facial_feature_detector as feature_detection
import panda_shot.PoseEstimation as PE
import tensorflow as tf
import panda_shot.network as network
import panda_shot.guided_filter as guided_filter
from tqdm import tqdm
from panda_shot.panda_pendant import panda_pendant
from panda_shot.panda_emoticons import panda_emoticons
from panda_shot.panda_environment import panda_environment
from panda_shot.panda_environmental_integration import Stylize_after_integration
from panda_shot.Stylization import Stylization
from panda_shot.panda_environmental_global import ReadMatchSuccess
from panda_shot.CartoonGAN import CartoonGAN
from panda_shot.RobustVideoMatting_master.inference import videoMatting
from panda_shot.WarpGAN_master.warpgan_test import warpgan_test
from panda_shot.UGATIT_master.main import selfie2anime

os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

this_path = os.path.dirname(os.path.abspath(__file__))
check.check_dlib_landmark_weights()
predictor_path_68f = "panda_shot/dlib_models/shape_predictor_68_face_landmarks.dat"
predictor_for_pose = dlib.shape_predictor(predictor_path_68f)
predictor_path_81f = 'panda_shot/dlib_models/shape_predictor_81_face_landmarks.dat'
predictor_for_lmarks = dlib.shape_predictor(predictor_path_81f)
detector_for_face = dlib.get_frontal_face_detector()




def demo_pose(request):
    # cap = cv2.VideoCapture(0)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
    # while (cap.isOpened()):
    #     x, y = demo_pose.demo(cap, out)
    #     print(x)
    #     print(y)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cap.release()
    #         out.release()
    #         cv2.destroyAllWindows()
    return render(request,'opencv_test.html')
    # return render(request,'opencv_offi.html')


def take_photo(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        photoClass = int(request.POST.getlist('photoClass')[0])
        personStyleKind = int(request.POST.getlist('personStyleKind')[0])
        envStyleKind = int(request.POST.getlist('envStyleKind')[0])
        videoMode = int(request.POST.getlist('videoMode')[0])
        personSegModel = int(request.POST.getlist('personSegModel')[0])
        envMode1_video = int(request.POST.getlist('envMode1_video')[0])

        '''
        print("photoClass:", photoClass)
        print("personStyleKind:", personStyleKind)
        print("envStyleKind:", envStyleKind)
        print("videoMode:", videoMode)
        print("personSegModel:", personSegModel)
        print("envMode1_video:", envMode1_video)
        '''

        imgArray = np.frombuffer(img.read(), np.uint8)
        frame = cv2.imdecode(imgArray, cv2.IMREAD_ANYCOLOR)

        '''
        if photoClass == 5:
            print("##### videoMatting #####")
            videoMatting()
            print("##### videoMatting over #####")
            return render(request, 'take_photo.html')
        '''

        #仅风格化流程
        if personStyleKind != -1 and envStyleKind != -1:
            print("---Only Stylize!!!---")
            if photoClass == 3 or photoClass == 4:
                if personStyleKind == 10 and envStyleKind == 10:#都是CartoonGAN
                    frame = CartoonGAN(frame)
                elif personStyleKind == 0 and envStyleKind == 0:#都是原图什么也不干
                    frame = frame
                else:
                    frame = Stylize_after_integration(personStyleKind, envStyleKind)
            else:
                if videoMode==0:
                    if personStyleKind != 10:
                        frame = cv2.resize(frame, (1200, 800), interpolation=cv2.INTER_CUBIC)
                        frame = Stylization(frame, personStyleKind)
                    else:
                        frame = CartoonGAN(frame)
        #非风格化
        else:
            if photoClass==2 or photoClass==3 or photoClass==4:
                frame, poseKind, type = panda_environment(frame, photoClass, personSegModel, envMode1_video)  # 人像分割环境融合
                #type=0匹配不成功；type=1匹配成功普通模式；type=2匹配成功且是视频模式
                if type == 0:
                    response1 = HttpResponse()
                    #匹配不成功，返回最近似的动作代号
                    response1['poseKind'] = poseKind
                    response1['envMode1_video'] = 0
                    return response1
                elif type == 2:
                    response1 = HttpResponse()
                    response1['poseKind'] = poseKind
                    response1['envMode1_video'] = 1
                    return response1
            elif photoClass==6: #warpgan
                frame = warpgan_test(frame, 5, 2.0)[0] #img, num_styles, scale
                #frame = cv2.resize(frame, (600, 450), interpolation=cv2.INTER_CUBIC)

            else:#人脸特征点识别
                dets = detector_for_face(frame, 0)
                # 检测特征点
                lmarks = feature_detection.get_landmarks(frame, detector_for_face, predictor_for_pose)
                if len(lmarks):
                    Pose_Para = PE.poseEstimation(frame, lmarks)
                    Pose_Angle = np.array(Pose_Para) * 180 / math.pi
                    for k, d in enumerate(dets):
                        shape = predictor_for_lmarks(frame, d)
                        # 返回一个2*81的landmarks数组元素，第一维x坐标，第二位y坐标
                        lmarks_cov_x = np.matrix([[p.x] for p in shape.parts()]).reshape(1, 81)
                        lmarks_cov_y = np.matrix([[p.y] for p in shape.parts()]).reshape(1, 81)
                        lmarks_cov = np.vstack((lmarks_cov_x, lmarks_cov_y))
                        # opencv画点
                        # for num in range(shape.num_parts):
                        #    cv2.circle(frame, (shape.parts()[num].x, shape.parts()[num].y), 3, (0, 255, 0), -1)
                        if photoClass == 0:
                            frame = panda_pendant(frame, lmarks_cov)  # 熊猫贴纸
                        elif photoClass == 1:
                            frame = panda_emoticons(frame, lmarks_cov)  # 熊猫头表情

                # print(Pose_Angle,lmarks_cov)
            # else:
            # print('No face detected!')

        #挂件加视频模式时实时风格化
        if personStyleKind != -1:
            if videoMode==1 and photoClass==0:
                if personStyleKind != 10:
                    frame = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_CUBIC)
                    frame = Stylization(frame, personStyleKind)
                else:
                    frame = CartoonGAN(frame)

        #返回二进制图片
        ret, buf = cv2.imencode(".jpg", frame)
        #return HttpResponse(buf.tobytes(), content_type='arraybuffer')
        response2 = HttpResponse(buf.tobytes(), content_type='arraybuffer')
        response2['poseKind'] = -1
        response2['envMode1_video'] = 0
        return response2
    return render(request, 'take_photo.html')

def InformMatchSuccess(request):
    #print("$$$$$    GETINFORM!!!")
    match_number = -1
    while 1:
        match_number = ReadMatchSuccess()
        #print("-------match_number: ", match_number)
        if match_number == -1:
            continue
        else:
            break
    #print("$$$$$    INFORMDONE!!!")
    response = HttpResponse()
    response['poseKind'] = match_number
    return response

def panda_video(request):
    if request.method == 'POST':
        videoFile = request.FILES.get('videoFile')
        videoBgNum = int(request.POST.getlist('videoBgNum')[0])
        poseKind = int(request.POST.getlist('poseKind')[0])

        #videoArray = np.frombuffer(videoFile.read(), np.uint8)
        #frame = cv2.imdecode(videoArray, cv2.IMREAD_ANYCOLOR)
        #print(videoFile)
        #print(type(videoFile))
        #print(sys.getsizeof(videoFile))
        #print("Save video!!!")
        with open('panda_shot/RobustVideoMatting_master/CapVideoForTest/input.mp4', 'wb+') as f:
            #f.write(videoFile)
            for chunk in videoFile.chunks():
                f.write(chunk)

        print("##### videoMatting #####")
        videoMatting(videoBgNum, poseKind)
        print("##### videoMatting done #####")
        '''
        video_path = 'panda_shot/RobustVideoMatting_master/CapVideoForTest/change.mp4'
        content = {'video_path': video_path}
        return render(request, 'take_photo_video.html', content)
        '''
        response = HttpResponse()
        response['video_path'] = "/static/images/change.mp4"
        return response
    return render(request, 'take_photo.html')


def WrapGAN(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        imgArray = np.frombuffer(img.read(), np.uint8)
        frame = cv2.imdecode(imgArray, cv2.IMREAD_ANYCOLOR)
        row, col, channel = frame.shape
        if row < col:
            frame = frame[ : , int((col-row)/2) : col-int((col-row)/2)]
        else:
            frame = frame[int((row - col) / 2) : row - int((row - col) / 2), : ]
        frame = cv2.resize(frame,(256,256))

        # ---selfie2anime
        selfie2anime_output =  selfie2anime(frame)

        ret, buf = cv2.imencode(".jpg", selfie2anime_output)
        # img_bytes = [buf.tobytes()]*5
        # img_sizes = str(len(buf.tobytes()))*5
        img_bytes = buf.tobytes()

        '''
        # ---WrapGAN
        num_styles = 5
        scale = 0.5
        warpgan_output = warpgan_test(frame, num_styles, scale)  # img, num_styles, scale

        img_bytes = []
        img_sizes = ''
        for i in range(num_styles):
            ret, buf = cv2.imencode(".jpg", warpgan_output[i])
            img_bytes.append(buf.tobytes())
            img_sizes = img_sizes + str(len(buf.tobytes()))
        '''

        response = HttpResponse(img_bytes, content_type='arraybuffer')
        #response['img_sizes'] = img_sizes
        return response
    return render(request, 'take_photo_wrapgan.html')