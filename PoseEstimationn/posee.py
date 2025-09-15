import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()

cap = cv2.VideoCapture(r"C:\Users\huzai\Desktop\Data Science\Opencv_practice\PoseEstimationn\2.mp4")

while True:
    success, img1 = cap.read()
    if not success:
        break

    img = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)

    clr_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    result  = pose.process(clr_img)

    if result.pose_landmarks:
        for id ,lm in enumerate(result.pose_landmarks.landmark):
            w,h,c = img1.shape
            cx,cy = int(lm.x * w ) , int(lm.y * h)
            cv2.circle(img,(cx,cy),7,(0,255,0),cv2.FILLED)

        mp_draw.draw_landmarks(img,result.pose_landmarks,mp_pose.POSE_CONNECTIONS)
            

    cv2.imshow("Showing video", img)

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
