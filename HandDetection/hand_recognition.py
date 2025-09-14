import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

while True:
    success , img = cam.read()

    if not success:
        break

    flip_img = cv2.flip(img,1)

    clr_img = cv2.cvtColor(flip_img,cv2.COLOR_BGR2RGB) 

    result=hands.process(clr_img)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:

            for id , lm in enumerate(hand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x * w) , int(lm.y * h)

                # print(id,cx,cy)
                if id == 4:
                    cv2.circle(flip_img,(cx,cy),15,(0,255,255),cv2.FILLED)


            mp_draw.draw_landmarks(flip_img,hand,mp_hands.HAND_CONNECTIONS)

            

    cv2.imshow("Video Showing",flip_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):  
        break

cam.release()
cv2.destroyAllWindows()