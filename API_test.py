import demo_pose
import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (1280, 720))
while(cap.isOpened()):
    x,y=demo_pose.demo(cap,out)
    print(x)
    print(y)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        out.release()
        cv2.destroyAllWindows()