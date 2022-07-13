import numpy as np
import math

#*****动作匹配方法
def match_pose2(kp, StandardPose):
    matchScore = 200 #匹配度阈值
    matchPose = -1
    for i in range(16):
        score = 0
        for j in range(10):
            if kp[j][0]!=0 or kp[j][1]!=0:
                score = score + getScore([kp[j][0]-kp[1][0],kp[j][1]-kp[1][1]], StandardPose[i][j]-StandardPose[i][1])
                #print(score)
        if kp[12][0]!=0 or kp[12][1]!=0:
            score = score + getScore([kp[j][0]-kp[1][0],kp[j][1]-kp[1][1]], StandardPose[i][j]-StandardPose[i][1])
        #print(score)
        if i==0:
            matchScore = score
            matchPose = i
        elif score < matchScore:  #匹配值越小越好
            matchScore = score
            matchPose = i
    print("**matchPose: " , matchPose)
    print("**matchScore: ", matchScore)
    perfectMatch = 0
    if matchScore <= 200:
        perfectMatch = 1
    else:
        perfectMatch = 0
    return matchPose + 2, perfectMatch


#*****综合动作匹配方法
def match_pose(kp):
    totalScoreThreshold = 30 #总匹配度阈值
    perfectMatch = 0  # 是否完美匹配

    matchPose = 0
    totalMatchScore_per_point = 0
    a = [0]*3
    b = [0]*3
    a[0], b[0] = match_pose_once(kp, StandardPose_0)
    a[1], b[1] = match_pose_once(kp, StandardPose_1)
    a[2], b[2] = match_pose_once(kp, StandardPose_2)

    #三个预测动作都不同，则取（总匹配值/总匹配点数）最小的
    if a[0] != a[1] and a[0] != a[2] and a[1] != a[2]:
        if b[0] <= b[1] and b[0] <= b[2]:
            matchPose = a[0]
            totalMatchScore_per_point = b[0]
        if b[1] <= b[0] and b[1] <= b[2]:
            matchPose = a[1]
            totalMatchScore_per_point = b[1]
        if b[2] <= b[1] and b[2] <= b[0]:
            matchPose = a[2]
            totalMatchScore_per_point = b[2]
    else:
        # 三个预测动作都相同，则取（总匹配值/总匹配点数）的平均数
        if a[0] == a[1] and a[0] == a[2]:
            matchPose = a[0]
            totalMatchScore_per_point = (b[0] + b[1] + b[2])/3
        else:
            # 三个预测动作中两个相同，则取这两个（总匹配值/总匹配点数）的平均数
            if a[0] == a[1]:
                matchPose = a[0]
                totalMatchScore_per_point = (b[0] + b[1])/2
            if a[0] == a[2]:
                matchPose = a[0]
                totalMatchScore_per_point = (b[0] + b[2])/2
            if a[1] == a[2]:
                matchPose = a[1]
                totalMatchScore_per_point = (b[1] + b[2])/2

    if totalMatchScore_per_point <= totalScoreThreshold:
        perfectMatch = 1
    else:
        perfectMatch = 0
    print("****MatchPose: ", matchPose, )
    print("****ScorePerP: ", totalMatchScore_per_point, )
    print("*****PerfectM: ", perfectMatch, )
    return matchPose + 2, perfectMatch
    #return 8, 1


#*****单次动作匹配方法
def match_pose_once(kp, StandardPose):
    singleScoreThreshold = 50 #单个点匹配度阈值
    totalMatchScore = 200
    totalMatchPointNum = 11
    matchPose = -1

    for i in range(16):
        score = 0
        totalScore = 0
        matchPointNum = 0 #匹配的关键点数目
        for j in range(10):
            if kp[j][0]!=0 or kp[j][1]!=0:
                score = getScore([kp[j][0]-kp[1][0],kp[j][1]-kp[1][1]], StandardPose[i][j]-StandardPose[i][1])
                totalScore = totalScore + score
                if score < singleScoreThreshold:
                    matchPointNum = matchPointNum + 1
        if kp[12][0]!=0 or kp[12][1]!=0:
            score = getScore([kp[j][0]-kp[1][0],kp[j][1]-kp[1][1]], StandardPose[i][j]-StandardPose[i][1])
            totalScore = totalScore + score
            if score < singleScoreThreshold:
                matchPointNum = matchPointNum + 1

        if i==0:
            totalMatchScore = totalScore
            totalMatchPointNum = matchPointNum
            matchPose = i
        elif (totalScore/matchPointNum) < (totalMatchScore/totalMatchPointNum) and matchPointNum > 5:  #匹配值越小越好
            totalMatchScore = totalScore
            totalMatchPointNum = matchPointNum
            matchPose = i
    #print("**matchPose: " , matchPose)
    #print("**matchScore: ", totalMatchScore)
    #print("**MatchPointNum: ", totalMatchPointNum)
    return matchPose, totalMatchScore/totalMatchPointNum



#*****计算两点之间的欧氏距离
def getScore(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))


#*****标准动作点集
StandardPose_0 = np.array([
[[3.08759613e+02, 1.32638885e+02, 9.07754779e-01],
[3.18566803e+02, 2.19522446e+02, 8.14746439e-01],
[2.42731415e+02, 2.19529526e+02, 6.12556458e-01],
[2.19459457e+02, 3.17413605e+02, 8.13909948e-01],
[1.60808014e+02, 3.57734985e+02, 7.89132893e-01],
[3.96922302e+02, 2.20677689e+02, 6.67359352e-01],
[4.34799316e+02, 3.18582581e+02, 7.92485654e-01],
[4.83726471e+02, 3.36983398e+02, 8.05760324e-01],
[3.10019257e+02, 4.21366150e+02, 2.78544456e-01],
[2.62323120e+02, 4.18911530e+02, 2.04777882e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[3.56527832e+02, 4.26248352e+02, 2.48823673e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[2.92880188e+02, 1.20365639e+02, 8.18026602e-01],
[3.25951477e+02, 1.16716545e+02, 8.44880879e-01],
[2.79434235e+02, 1.24059624e+02, 7.22628355e-01],
[3.56462250e+02, 1.22808731e+02, 9.15137351e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

[[3.4184979e+02, 1.0696155e+02, 8.1744075e-01],
[3.8342688e+02, 1.8403505e+02, 6.7621583e-01],
[3.0756265e+02, 1.8279745e+02, 5.2449083e-01],
[1.9991637e+02, 1.9012047e+02, 7.5292349e-01],
[9.3522667e+01, 1.6081194e+02, 7.9328394e-01],
[4.6540573e+02, 1.9012135e+02, 5.3839481e-01],
[5.7309967e+02, 1.8158858e+02, 8.2319456e-01],
[5.1311926e+02, 9.3500725e+01, 7.6143104e-01],
[3.7855682e+02, 4.0790323e+02, 1.6501571e-01],
[3.2591855e+02, 4.1523004e+02, 1.7071725e-01],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[4.3480026e+02, 4.1157077e+02, 1.5657122e-01],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[3.3936926e+02, 9.4717125e+01, 8.3120275e-01],
[3.5897314e+02, 9.2296768e+01, 8.9172673e-01],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[4.0792868e+02, 1.0446747e+02, 8.5519868e-01],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
[0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],

[[220.7738, 83.69339, 0.87641394],
[191.38521, 163.24191, 0.51681066],
[159.54321, 154.64603, 0.5543368],
[286.79855, 160.78444, 0.5463526],
[440.93732, 149.71663, 0.5435406],
[225.6164, 170.57956, 0.21628322],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[201.13791, 390.7769, 0.23286511],
[225.6179, 388.32507, 0.16446055],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[166.89215, 399.33377, 0.15460286],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[202.35019, 71.42777, 0.8312811],
[232.95903, 72.63376, 0.85615647],
[168.12573, 82.523575, 0.86090136],
[246.3761, 77.57556, 0.09108202],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0],
[0.0, 0.0, 0.0]],


[[ 279.5,108.2,0.8 ],
[ 307.6,176.7,0.4 ],
[ 288.0,182.8,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 328.4,173.0,0.4 ],
[ 196.3,110.6,0.7 ],
[ 114.3,72.7,0.8 ],
[ 268.4,404.2,0.3 ],
[ 240.3,404.2,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 297.8,403.0,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 279.5,97.2,0.7 ],
[ 300.3,94.7,0.8 ],
[ 0.0,0.0,0.0 ],
[ 339.4,121.6,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[2.98989990e+02, 1.22870346e+02, 9.31365848e-01],
[2.99055603e+02, 1.91398361e+02, 7.24641323e-01],
[2.26825455e+02, 1.93817520e+02, 5.34960091e-01],
[1.41185379e+02, 1.61959747e+02, 8.49011779e-01],
[1.77907990e+02, 8.24270172e+01, 7.48549163e-01],
[3.76079529e+02, 1.90168243e+02, 6.20039344e-01],
[4.66663635e+02, 1.62040619e+02, 8.19603026e-01],
[4.28718140e+02, 7.75863953e+01, 7.34628320e-01],
[3.01475067e+02, 4.12792358e+02, 2.27091044e-01],
[2.54964828e+02, 4.11566864e+02, 1.81906000e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[3.54074524e+02, 4.06683319e+02, 1.87230110e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[2.84355286e+02, 1.13020203e+02, 8.84061396e-01],
[3.13707672e+02, 1.11860062e+02, 8.72260392e-01],
[2.64774384e+02, 1.22896683e+02, 8.20607841e-01],
[3.36900604e+02, 1.22831375e+02, 9.21481133e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

[[2.79455902e+02, 1.31356369e+02, 9.10444498e-01],
[2.95340973e+02, 1.90138885e+02, 7.30035305e-01],
[2.20719833e+02, 1.91394363e+02, 5.39394140e-01],
[1.33886887e+02, 1.60792191e+02, 8.06191325e-01],
[8.37338867e+01, 9.22981949e+01, 8.19921374e-01],
[3.67548096e+02, 1.85256744e+02, 5.89245975e-01],
[4.55621307e+02, 1.60801895e+02, 8.25183153e-01],
[5.14370911e+02, 8.73466492e+01, 8.23578954e-01],
[3.00267029e+02, 3.90781372e+02, 2.39476308e-01],
[2.51328323e+02, 3.94448700e+02, 1.96961954e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[3.50393494e+02, 3.95664459e+02, 1.91958159e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[2.68384888e+02, 1.20349014e+02, 8.73911023e-01],
[2.96529083e+02, 1.14279251e+02, 8.43166113e-01],
[2.53767914e+02, 1.22902618e+02, 3.14558953e-01],
[3.24718628e+02, 1.21657700e+02, 8.94834995e-01],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
[0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

[[ 253.75772,91.042694,0.82928616 ],
[ 213.38602,180.31253,0.6593274 ],
[ 180.34406,165.67935,0.58803105 ],
[ 281.8987,272.11072,0.6347234 ],
[ 374.88174,209.73268,0.73917556 ],
[ 248.8449,192.56555,0.51229405 ],
[ 306.36536,276.98294,0.38825822 ],
[ 376.10596,215.82301,0.34983784 ],
[ 267.1929,385.8849,0.20859587 ],
[ 257.41208,388.3278,0.19177671 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 281.88028,378.5407,0.1748774 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 239.02776,76.35339,0.8356409 ],
[ 258.68793,78.80779,0.8350078 ],
[ 191.34912,89.80443,0.85264707 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[2.9774509e+02, 9.3462051e+01, 8.9346397e-01],
 [2.8067773e+02, 1.9870125e+02, 7.1539801e-01],
 [2.1336543e+02, 1.8281775e+02, 5.4530364e-01],
 [2.6109940e+02, 3.0633176e+02, 6.3130593e-01],
 [3.5895352e+02, 2.9906918e+02, 5.9235775e-01],
 [3.4792563e+02, 2.1094470e+02, 6.6912895e-01],
 [3.8715768e+02, 3.1861954e+02, 7.5247920e-01],
 [2.7577267e+02, 3.1858560e+02, 2.7576163e-01],
 [2.9411548e+02, 4.0423975e+02, 3.7741891e-01],
 [2.5620682e+02, 4.0790671e+02, 3.2092944e-01],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [3.3327710e+02, 4.0300739e+02, 3.4072274e-01],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [2.7817587e+02, 8.1236038e+01, 8.5633695e-01],
 [3.1246161e+02, 7.8803406e+01, 7.8897929e-01],
 [2.4638806e+02, 9.8388138e+01, 8.1772447e-01],
 [3.2710895e+02, 8.9826912e+01, 1.6307995e-01],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
 [0.0000000e+00, 0.0000000e+00, 0.0000000e+00]],


[[ 387.0756,114.28153,0.8907261 ],
[ 381.0007,207.2446,0.662706 ],
[ 291.66986,193.81313,0.47896722 ],
[ 206.04706,141.15524,0.7055822 ],
[ 121.6324,99.59597,0.7221912 ],
[ 464.1755,218.24707,0.65018064 ],
[ 525.3459,299.0727,0.7978471 ],
[ 483.69315,367.52325,0.78770196 ],
[ 356.51245,418.91556,0.113377824 ],
[ 291.67896,395.6639,0.07771323 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 416.4668,427.47418,0.14146383 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 368.7673,103.31719,0.86408406 ],
[ 404.21014,103.305626,0.8784039 ],
[ 346.69543,122.84661,0.89967793 ],
[ 423.80377,122.88751,0.85747683 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 170.5,91.0,0.9 ],
[ 141.2,170.6,0.7 ],
[ 100.8,162.0,0.6 ],
[ 188.9,251.3,0.5 ],
[ 308.8,328.4,0.7 ],
[ 180.3,175.4,0.6 ],
[ 258.7,139.9,0.8 ],
[ 327.1,70.2,0.8 ],
[ 176.7,394.4,0.1 ],
[ 154.7,399.3,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 202.4,374.9,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 151.0,75.2,0.9 ],
[ 180.4,81.2,0.9 ],
[ 111.8,93.5,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[2.70902832e+02, 1.36318237e+02, 8.29290509e-01],
  [2.85540710e+02, 2.08476196e+02, 7.33781576e-01],
  [2.20717636e+02, 2.10934525e+02, 6.89795256e-01],
  [1.32664871e+02, 2.25602936e+02, 7.74322569e-01],
  [6.16628342e+01, 2.30527145e+02, 8.23118985e-01],
  [3.49170868e+02, 2.02381165e+02, 5.87064743e-01],
  [4.50712494e+02, 2.08455765e+02, 7.48091578e-01],
  [5.35134583e+02, 2.10905045e+02, 8.46001565e-01],
  [2.91657715e+02, 4.16468842e+02, 4.02364731e-01],
  [2.53765686e+02, 4.16475830e+02, 3.57511938e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [3.30830414e+02, 4.16480713e+02, 3.43908519e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [2.58666992e+02, 1.28958252e+02, 8.74513507e-01],
  [2.87992554e+02, 1.24087875e+02, 8.65859866e-01],
  [2.46426468e+02, 1.42416245e+02, 7.48025596e-01],
  [3.11255951e+02, 1.33871140e+02, 8.25395882e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],

[[ 239.03491,90.99962,0.872705 ],
[ 236.61066,175.44173,0.6931453 ],
[ 170.53871,180.37209,0.66630954 ],
[ 72.72884,217.05914,0.8114506 ],
[ 46.970005,130.18678,0.77468 ],
[ 300.23764,170.57123,0.5354961 ],
[ 407.9354,124.080376,0.6498981 ],
[ 505.78943,66.56837,0.75167 ],
[ 240.31859,392.00107,0.3587361 ],
[ 197.4563,393.21857,0.32612813 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 287.97504,389.56152,0.32397527 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 221.94647,76.34206,0.8498404 ],
[ 254.9672,77.59471,0.8090776 ],
[ 201.13126,91.02535,0.8418424 ],
[ 277.00818,92.24,0.88469785 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 219.43477,113.025894,0.9002622 ],
[ 199.87137,202.393,0.63530725 ],
[ 160.75345,195.02313,0.53580433 ],
[ 261.11047,285.53824,0.74101543 ],
[ 382.22495,367.52618,0.71077967 ],
[ 234.1701,212.15782,0.49371848 ],
[ 317.39096,259.88702,0.74791086 ],
[ 417.6908,302.68448,0.77429205 ],
[ 215.82451,415.2368,0.11470046 ],
[ 174.23376,417.6967,0.10802891 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 254.98099,407.89664,0.09722923 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 197.44899,100.81808,0.80304515 ],
[ 229.30017,99.56267,0.8632823 ],
[ 161.98645,115.52962,0.85725784 ],
[ 240.32736,104.488304,0.080513656 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[ 258.6594,110.58732,0.886269 ],
[ 279.45773,201.15697,0.8352094 ],
[ 198.71019,210.94574,0.63104117 ],
[ 170.49365,322.26675,0.716149 ],
[ 236.64062,203.5881,0.7247084 ],
[ 363.86603,187.67403,0.5494058 ],
[ 367.5105,279.43768,0.6656643 ],
[ 262.32184,190.11832,0.39478868 ],
[ 281.9086,434.80927,0.22241761 ],
[ 225.60663,436.02832,0.18597291 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 334.48672,433.59445,0.18683182 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 232.95331,102.08361,0.84731853 ],
[ 269.62433,86.15892,0.852631 ],
[ 214.59769,121.616035,0.75959945 ],
[ 297.7956,92.2577,0.895056 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[ 319.8,76.4,0.8 ],
[ 383.4,182.8,0.6 ],
[ 301.5,190.1,0.4 ],
[ 182.8,181.6,0.8 ],
[ 141.2,91.0,0.8 ],
[ 455.6,179.1,0.5 ],
[ 395.7,248.9,0.6 ],
[ 294.1,171.8,0.7 ],
[ 339.4,423.8,0.3 ],
[ 296.6,421.4,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 387.1,432.4,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 325.9,65.3,0.2 ],
[ 340.6,62.8,0.8 ],
[ 0.0,0.0,0.0 ],
[ 390.8,80.0,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[2.84326660e+02, 1.19160126e+02, 8.00121069e-01],
  [2.97806885e+02, 2.08462418e+02, 6.92332208e-01],
  [2.10905884e+02, 2.10928131e+02, 5.38512528e-01],
  [1.05732201e+02, 2.78192078e+02, 7.19116390e-01],
  [1.27744820e+02, 3.46707062e+02, 7.65862942e-01],
  [3.89537964e+02, 2.01137894e+02, 4.95435059e-01],
  [4.96007812e+02, 2.63535828e+02, 7.10053205e-01],
  [4.74009491e+02, 3.49184509e+02, 7.80897200e-01],
  [3.05115021e+02, 4.37278595e+02, 1.93775818e-01],
  [2.53745438e+02, 4.37265717e+02, 1.45156279e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [3.49184540e+02, 4.38487549e+02, 1.60265505e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [2.63530396e+02, 1.03251030e+02, 8.58574986e-01],
  [3.06353882e+02, 9.83763809e+01, 8.32499981e-01],
  [2.43976166e+02, 1.21658112e+02, 6.15049839e-01],
  [3.38169250e+02, 1.11880356e+02, 9.09193873e-01],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
  [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]
], dtype='int64')



StandardPose_1 = np.array([
[[ 318.6,130.2,0.9 ],
[ 317.4,240.3,0.7 ],
[ 229.2,241.6,0.6 ],
[ 208.5,368.8,0.7 ],
[ 151.0,374.8,0.7 ],
[ 409.1,241.5,0.6 ],
[ 427.5,372.4,0.7 ],
[ 496.0,346.7,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 299.0,113.1,0.9 ],
[ 337.0,111.8,0.9 ],
[ 270.9,130.2,0.8 ],
[ 366.3,122.9,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 401.8,122.9,0.9 ],
[ 388.3,219.5,0.7 ],
[ 308.8,214.6,0.5 ],
[ 164.4,182.8,0.6 ],
[ 38.4,142.4,0.6 ],
[ 467.8,223.2,0.6 ],
[ 590.2,191.4,0.8 ],
[ 529.0,96.0,0.7 ],
[ 373.6,447.1,0.1 ],
[ 321.0,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 426.2,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 379.8,110.6,0.8 ],
[ 418.9,109.4,0.8 ],
[ 349.2,127.7,0.8 ],
[ 442.2,122.9,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 231.8,130.2,0.9 ],
[ 179.1,215.8,0.2 ],
[ 152.2,208.5,0.4 ],
[ 212.2,186.5,0.1 ],
[ 0.0,0.0,0.0 ],
[ 198.7,226.8,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 209.7,111.8,0.9 ],
[ 241.5,115.5,0.8 ],
[ 164.4,128.9,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 305.1,100.8,0.8 ],
[ 316.1,174.2,0.3 ],
[ 269.6,197.5,0.3 ],
[ 190.1,177.9,0.7 ],
[ 108.2,130.2,0.7 ],
[ 347.9,162.0,0.4 ],
[ 197.5,102.0,0.7 ],
[ 102.0,62.9,0.7 ],
[ 270.9,409.1,0.2 ],
[ 241.5,404.2,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 300.3,415.2,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 294.1,88.6,0.8 ],
[ 327.1,84.9,0.9 ],
[ 280.7,102.0,0.1 ],
[ 359.0,111.8,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],



[[ 305.1,132.6,0.9 ],
[ 300.3,228.1,0.7 ],
[ 224.4,223.2,0.5 ],
[ 111.8,153.4,0.7 ],
[ 186.5,56.8,0.7 ],
[ 377.3,229.3,0.5 ],
[ 497.2,173.0,0.8 ],
[ 437.3,74.0,0.8 ],
[ 297.8,447.1,0.1 ],
[ 247.6,447.1,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 346.7,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 286.7,121.6,0.9 ],
[ 322.3,119.2,0.8 ],
[ 259.9,140.0,0.9 ],
[ 346.8,138.7,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 297.8,152.3,0.9 ],
[ 299.0,239.0,0.7 ],
[ 230.5,235.4,0.6 ],
[ 122.8,173.0,0.7 ],
[ 66.6,73.9,0.7 ],
[ 367.5,239.1,0.6 ],
[ 475.2,160.7,0.8 ],
[ 532.7,62.9,0.9 ],
[ 294.1,447.1,0.1 ],
[ 247.6,447.1,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 344.3,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 279.5,141.2,0.9 ],
[ 316.1,141.1,0.9 ],
[ 259.9,153.5,0.8 ],
[ 340.6,152.2,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 284.3,124.1,0.8 ],
[ 210.9,209.7,0.4 ],
[ 180.3,190.1,0.5 ],
[ 292.9,328.4,0.6 ],
[ 360.2,229.3,0.7 ],
[ 245.2,230.5,0.2 ],
[ 288.0,336.9,0.1 ],
[ 0.0,0.0,0.0 ],
[ 226.8,437.3,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 268.5,103.2,0.9 ],
[ 288.0,109.4,0.8 ],
[ 209.7,109.4,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 306.4,113.0,0.9 ],
[ 275.8,220.7,0.7 ],
[ 212.2,206.0,0.5 ],
[ 277.0,359.0,0.5 ],
[ 356.5,329.6,0.3 ],
[ 328.4,235.4,0.7 ],
[ 405.5,328.4,0.8 ],
[ 328.4,307.5,0.7 ],
[ 270.9,447.1,0.1 ],
[ 224.4,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 314.9,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 279.4,95.9,0.8 ],
[ 318.6,100.8,0.9 ],
[ 239.1,115.5,0.9 ],
[ 329.7,104.5,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 379.8,131.4,0.9 ],
[ 382.2,233.0,0.7 ],
[ 302.7,221.9,0.6 ],
[ 180.3,154.6,0.7 ],
[ 92.3,75.1,0.8 ],
[ 464.2,246.4,0.6 ],
[ 544.9,328.4,0.8 ],
[ 494.7,388.3,0.7 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 362.6,115.5,0.8 ],
[ 399.4,113.1,0.9 ],
[ 338.1,133.9,0.8 ],
[ 426.3,131.4,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 184.0,111.8,0.9 ],
[ 135.1,193.8,0.4 ],
[ 110.6,195.0,0.4 ],
[ 267.2,306.3,0.6 ],
[ 422.6,381.0,0.6 ],
[ 163.2,193.8,0.3 ],
[ 233.0,152.2,0.8 ],
[ 306.4,83.7,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 162.0,92.2,0.9 ],
[ 196.3,98.4,0.8 ],
[ 111.9,103.3,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 305.1,161.9,0.9 ],
[ 307.5,247.6,0.8 ],
[ 239.0,241.5,0.6 ],
[ 121.6,228.0,0.8 ],
[ 16.4,209.7,0.8 ],
[ 377.3,248.9,0.6 ],
[ 492.3,236.6,0.6 ],
[ 592.6,223.2,0.4 ],
[ 301.5,447.1,0.3 ],
[ 259.9,447.1,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 345.5,447.1,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 288.0,151.0,0.9 ],
[ 319.8,148.5,0.8 ],
[ 268.4,169.3,0.9 ],
[ 345.5,163.2,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 259.9,121.6,0.9 ],
[ 267.2,223.2,0.7 ],
[ 184.0,228.1,0.5 ],
[ 59.2,263.5,0.7 ],
[ 32.3,147.3,0.7 ],
[ 347.9,218.3,0.4 ],
[ 460.5,132.7,0.7 ],
[ 551.1,62.9,0.6 ],
[ 270.9,447.1,0.1 ],
[ 220.7,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 324.7,434.8,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 241.5,104.5,0.9 ],
[ 280.7,103.3,0.9 ],
[ 220.8,122.9,0.8 ],
[ 312.5,122.8,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 212.1,120.4,0.8 ],
[ 170.6,219.5,0.7 ],
[ 121.6,201.1,0.5 ],
[ 245.2,314.9,0.7 ],
[ 398.1,379.8,0.7 ],
[ 217.0,234.2,0.5 ],
[ 313.7,274.5,0.7 ],
[ 427.5,316.1,0.8 ],
[ 195.0,434.8,0.1 ],
[ 153.4,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 232.9,428.7,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 191.3,102.0,0.9 ],
[ 220.7,103.3,0.9 ],
[ 141.2,121.6,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 259.9,116.7,0.9 ],
[ 272.1,219.5,0.8 ],
[ 186.5,230.5,0.6 ],
[ 160.8,348.0,0.7 ],
[ 229.3,211.0,0.8 ],
[ 362.6,201.1,0.6 ],
[ 398.1,267.2,0.7 ],
[ 250.1,181.5,0.3 ],
[ 288.0,447.1,0.1 ],
[ 231.7,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 335.7,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 236.6,108.1,0.8 ],
[ 277.0,93.5,0.9 ],
[ 209.7,131.4,0.8 ],
[ 305.1,105.7,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 327.2,103.3,0.9 ],
[ 376.1,220.7,0.6 ],
[ 302.7,218.3,0.4 ],
[ 171.8,190.1,0.8 ],
[ 137.5,86.1,0.7 ],
[ 464.2,212.2,0.5 ],
[ 396.9,268.4,0.6 ],
[ 297.8,173.0,0.7 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 332.1,93.5,0.1 ],
[ 349.2,88.6,0.8 ],
[ 0.0,0.0,0.0 ],
[ 396.9,111.9,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 299.0,111.8,0.9 ],
[ 300.2,224.4,0.7 ],
[ 209.7,225.6,0.5 ],
[ 92.3,279.4,0.5 ],
[ 105.7,307.6,0.4 ],
[ 393.2,226.8,0.5 ],
[ 514.3,288.0,0.6 ],
[ 494.8,318.6,0.6 ],
[ 294.1,447.1,0.1 ],
[ 239.1,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 340.6,447.1,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 278.3,97.1,0.9 ],
[ 318.6,93.6,0.9 ],
[ 252.5,115.5,0.8 ],
[ 350.4,113.0,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]]
], dtype='int64')




StandardPose_2 = np.array([
[[ 305.1,98.4,0.8 ],
[ 306.4,188.9,0.8 ],
[ 233.0,190.1,0.7 ],
[ 209.6,289.3,0.8 ],
[ 144.8,346.7,0.8 ],
[ 376.1,188.9,0.7 ],
[ 396.9,289.2,0.8 ],
[ 466.6,318.6,0.8 ],
[ 308.8,383.4,0.5 ],
[ 268.4,385.9,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 348.0,378.6,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 288.0,84.9,0.9 ],
[ 318.6,83.7,0.9 ],
[ 268.4,99.6,0.8 ],
[ 339.4,93.5,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 373.6,88.6,0.8 ],
[ 387.2,171.8,0.8 ],
[ 325.9,169.3,0.6 ],
[ 214.6,160.8,0.8 ],
[ 104.5,148.5,0.7 ],
[ 453.2,175.4,0.7 ],
[ 546.2,171.8,0.8 ],
[ 509.4,100.8,0.8 ],
[ 388.3,387.1,0.4 ],
[ 346.7,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 429.9,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 361.4,82.4,0.9 ],
[ 389.5,75.2,0.8 ],
[ 354.1,93.5,0.1 ],
[ 423.8,93.5,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 231.8,93.6,0.9 ],
[ 201.1,162.0,0.3 ],
[ 192.6,163.2,0.6 ],
[ 336.9,153.5,0.7 ],
[ 454.4,166.9,0.5 ],
[ 210.9,154.7,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 224.4,371.2,0.3 ],
[ 219.5,374.9,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 237.8,367.5,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 213.4,83.7,0.9 ],
[ 239.1,83.6,0.9 ],
[ 179.1,99.6,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 289.2,94.7,0.8 ],
[ 334.5,160.8,0.4 ],
[ 334.5,168.1,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 341.8,152.2,0.5 ],
[ 210.9,102.0,0.7 ],
[ 122.8,70.2,0.8 ],
[ 277.0,372.4,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 290.5,86.1,0.1 ],
[ 306.4,83.6,0.9 ],
[ 0.0,0.0,0.0 ],
[ 345.5,102.0,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 301.4,97.2,0.8 ],
[ 305.1,173.0,0.8 ],
[ 240.3,171.8,0.7 ],
[ 135.1,149.7,0.8 ],
[ 181.5,64.1,0.8 ],
[ 367.5,173.0,0.6 ],
[ 469.1,141.2,0.8 ],
[ 426.3,67.8,0.8 ],
[ 302.7,381.0,0.5 ],
[ 259.9,381.0,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 340.6,381.0,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 288.0,86.2,0.8 ],
[ 318.5,84.9,0.9 ],
[ 269.7,100.8,0.8 ],
[ 339.4,99.6,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 284.3,99.6,0.8 ],
[ 288.0,171.8,0.8 ],
[ 225.6,171.8,0.6 ],
[ 125.3,141.2,0.8 ],
[ 74.0,62.8,0.8 ],
[ 350.4,171.8,0.6 ],
[ 453.2,142.4,0.8 ],
[ 514.4,72.7,0.9 ],
[ 288.0,377.3,0.5 ],
[ 246.4,377.3,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 327.2,376.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 268.4,89.8,0.8 ],
[ 299.0,86.1,0.9 ],
[ 250.1,103.3,0.8 ],
[ 319.9,99.6,0.8 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 262.3,97.2,0.8 ],
[ 230.5,177.9,0.7 ],
[ 203.6,162.0,0.6 ],
[ 280.7,268.4,0.7 ],
[ 361.4,212.1,0.8 ],
[ 257.4,191.4,0.5 ],
[ 307.5,274.6,0.5 ],
[ 363.9,217.0,0.3 ],
[ 245.2,367.5,0.3 ],
[ 239.1,370.0,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 258.7,365.1,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 246.4,84.9,0.8 ],
[ 269.7,87.3,0.9 ],
[ 209.7,99.6,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 297.8,93.5,0.9 ],
[ 289.3,182.8,0.8 ],
[ 230.5,174.2,0.6 ],
[ 281.9,268.4,0.7 ],
[ 360.2,267.2,0.7 ],
[ 341.8,191.4,0.7 ],
[ 385.9,280.7,0.8 ],
[ 316.2,311.2,0.7 ],
[ 301.5,378.5,0.5 ],
[ 268.4,381.0,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 335.7,377.3,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 279.4,83.6,0.9 ],
[ 308.8,82.5,0.9 ],
[ 250.1,99.6,0.9 ],
[ 321.0,92.2,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 371.2,89.8,0.8 ],
[ 383.4,171.8,0.8 ],
[ 318.6,168.1,0.6 ],
[ 212.2,124.1,0.7 ],
[ 132.7,81.2,0.8 ],
[ 450.7,179.1,0.7 ],
[ 520.5,259.9,0.8 ],
[ 475.2,346.7,0.8 ],
[ 365.1,382.2,0.5 ],
[ 325.9,378.6,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 405.4,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 356.5,81.2,0.9 ],
[ 387.1,75.1,0.9 ],
[ 340.6,93.5,0.7 ],
[ 414.0,91.0,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 191.4,93.5,0.9 ],
[ 168.1,162.0,0.7 ],
[ 144.9,165.7,0.6 ],
[ 231.7,258.7,0.8 ],
[ 328.4,340.6,0.8 ],
[ 191.3,158.3,0.6 ],
[ 250.1,121.6,0.8 ],
[ 316.2,64.2,0.8 ],
[ 191.3,368.8,0.3 ],
[ 181.6,371.2,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 204.8,366.3,0.2 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 173.0,82.5,0.9 ],
[ 200.0,82.5,0.9 ],
[ 141.1,94.7,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 289.2,102.0,0.9 ],
[ 297.8,179.1,0.8 ],
[ 233.0,180.3,0.7 ],
[ 133.9,193.8,0.8 ],
[ 51.9,208.5,0.8 ],
[ 363.8,179.1,0.7 ],
[ 464.2,190.1,0.8 ],
[ 553.5,200.0,0.8 ],
[ 296.5,374.9,0.5 ],
[ 257.4,374.9,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 332.1,373.6,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 275.8,92.3,0.9 ],
[ 303.9,89.8,0.8 ],
[ 259.8,109.4,0.8 ],
[ 327.2,102.1,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 250.0,92.2,0.9 ],
[ 250.1,168.1,0.8 ],
[ 185.2,173.0,0.7 ],
[ 97.2,220.7,0.8 ],
[ 61.7,142.4,0.8 ],
[ 318.6,162.0,0.6 ],
[ 407.9,119.2,0.7 ],
[ 486.2,83.7,0.8 ],
[ 258.6,376.1,0.5 ],
[ 218.3,376.1,0.5 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 297.7,376.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 232.9,82.4,0.9 ],
[ 264.8,81.2,0.8 ],
[ 215.8,97.2,0.8 ],
[ 286.8,93.4,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 199.9,92.2,0.9 ],
[ 190.1,173.0,0.7 ],
[ 147.3,162.0,0.6 ],
[ 237.8,257.4,0.7 ],
[ 335.7,329.6,0.8 ],
[ 229.3,184.0,0.7 ],
[ 291.7,228.0,0.8 ],
[ 374.9,273.3,0.7 ],
[ 210.9,370.0,0.3 ],
[ 185.2,372.4,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 237.9,367.5,0.3 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 177.9,82.5,0.8 ],
[ 208.5,77.6,0.9 ],
[ 150.9,102.0,0.9 ],
[ 219.5,83.7,0.1 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 264.7,92.2,0.9 ],
[ 281.9,175.5,0.7 ],
[ 215.8,187.7,0.7 ],
[ 191.3,286.8,0.8 ],
[ 230.5,177.9,0.8 ],
[ 359.0,163.2,0.6 ],
[ 344.3,229.3,0.7 ],
[ 253.7,162.0,0.8 ],
[ 288.0,387.1,0.5 ],
[ 242.7,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 329.6,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 246.4,83.7,0.9 ],
[ 277.0,73.9,0.9 ],
[ 230.5,104.5,0.8 ],
[ 301.5,82.5,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 325.9,86.1,0.8 ],
[ 359.0,174.2,0.7 ],
[ 291.7,181.6,0.5 ],
[ 199.9,170.6,0.8 ],
[ 149.7,103.3,0.8 ],
[ 417.7,170.5,0.6 ],
[ 383.5,239.1,0.8 ],
[ 300.3,177.9,0.8 ],
[ 332.1,382.2,0.5 ],
[ 291.7,377.3,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 368.8,385.9,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 318.6,81.3,0.8 ],
[ 344.3,75.2,0.8 ],
[ 0.0,0.0,0.0 ],
[ 384.7,93.5,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]],


[[ 299.0,84.9,0.9 ],
[ 302.7,171.8,0.8 ],
[ 226.8,171.8,0.6 ],
[ 141.1,239.0,0.8 ],
[ 131.4,317.3,0.8 ],
[ 376.1,170.6,0.7 ],
[ 470.3,230.5,0.7 ],
[ 472.8,306.3,0.8 ],
[ 301.5,384.7,0.4 ],
[ 258.7,385.9,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 344.3,387.1,0.4 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 283.1,72.7,0.9 ],
[ 317.4,72.7,0.9 ],
[ 263.6,86.1,0.8 ],
[ 340.6,83.7,0.9 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ],
[ 0.0,0.0,0.0 ]]
], dtype='int64')

