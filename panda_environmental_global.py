MatchSuccess = 0
MatchNumber = 0
Mutex = 0

def WriteMatchSuccess(match_number):
    global MatchSuccess
    global MatchNumber
    global Mutex
    while 1:
        if Mutex == 0:
            Mutex = 1
            MatchSuccess = 1
            MatchNumber = match_number
            #print("--------Write Success !!!")
            Mutex = 0
            break
        else:
            continue

def ReadMatchSuccess():
    global MatchSuccess
    global MatchNumber
    global Mutex
    match_number = -1
    while 1:
        if Mutex == 0:
            Mutex = 1
            if MatchSuccess == 1:
                MatchSuccess = 0
                match_number = MatchNumber
                #print("--------Read Success !!!")
            Mutex = 0
            break
        else:
            continue
    return match_number