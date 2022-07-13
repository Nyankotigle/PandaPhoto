import cv2
from PIL import Image
import numpy as np
import argparse




def Stylization(frame, model_num):
    model = "panda_shot/Style_model/"
    if model_num == 0:
        return frame
    if model_num == 1:
        model = model + "candy.t7"
    if model_num == 2:
        model = model + "composition_vii.t7"
    if model_num == 3:
        model = model + "feathers.t7"
    if model_num == 4:
        model = model + "la_muse.t7"
    if model_num == 5:
        model = model + "mosaic.t7"
    if model_num == 6:
        model = model + "starry_night.t7"
    if model_num == 7:
        model = model + "the_scream.t7"
    if model_num == 8:
        model = model + "the_wave.t7"
    if model_num == 9:
        model = model + "udnie.t7"

    net = cv2.dnn.readNetFromTorch(model)
    inWidth = frame.shape[1]
    inHeight = frame.shape[0]
    inp = cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),(103.939, 116.779, 123.68), swapRB=False, crop=False)

    net.setInput(inp)
    out = net.forward()

    out = out.reshape(3, out.shape[2], out.shape[3])
    out[0] += 103.939
    out[1] += 116.779
    out[2] += 123.68
    #out /= 255
    out = out.transpose(1, 2, 0)
    out = out.astype(np.int)#转换为整数
    """
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    print (t / freq, 'ms')
    """

    #print(out.shape, out)
    return out