__author__ = 'Chenyu and Liuyang'

import dlib
import os
import numpy as np
import matplotlib.pyplot as plt

def _shape_to_np(shape):
    xy = []
    for i in range(68):
        xy.append((shape.part(i).x, shape.part(i).y,))
    xy = np.asarray(xy, dtype='float32')
    return xy
def get_lmarks(img,shape,dets,PlotOn=False):
    lmarks=[]
    if len(dets) > 0:
        for k, det in enumerate(dets):
            xy = _shape_to_np(shape)
            lmarks.append(xy)
        lmarks = np.asarray(lmarks, dtype='float32')
        lmarks = lmarks[0,:,:].T
        lmarks = np.squeeze(lmarks)
        if PlotOn:
            display_landmarks(img, lmarks)
        return lmarks
    else:
        return lmarks
def get_landmarks(img, detector, predictor, PlotOn=False):
    lmarks = []
    dets, scores, idx = detector.run(img, 1)
    # dets = [dlib.rectangle(left=0, top=0, right=img.shape[1], bottom=img.shape[0])]
    print("Number of faces detected: {}".format(len(dets)))
    if len(dets) > 0:
        for k, det in enumerate(dets):
            shape = predictor(img, det)
            xy = _shape_to_np(shape)
            lmarks.append(xy)
        lmarks = np.asarray(lmarks, dtype='float32')
        lmarks = lmarks[0,:,:].T
        lmarks = np.squeeze(lmarks)
        if PlotOn:
            display_landmarks(img, lmarks)
        return lmarks
    else:
        return lmarks

def display_landmarks(img, lmarks):
    for i in range(68):  
        xy = lmarks[:, i]  
        plt.plot(xy[0], xy[1], 'ro')  
        plt.text(xy[0], xy[1], str(i))
    plt.imshow(img)  
    plt.show()  