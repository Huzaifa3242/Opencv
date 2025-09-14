import cv2
import mediapipe as mp
import time



class HandRecognition():

    def __init__(self,static_mode=False,no_hands=2,det_confidence=0.5,track_confidence = 0.5):
        self.static_mode = static_mode
        self.no_hands= no_hands
        self.det_confidence=det_confidence
        self.track_confidence = track_confidence
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hand_landmarks = self.mp_hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.no_hands,
            min_detection_confidence=self.det_confidence,
            min_tracking_confidence=self.track_confidence
        )

    def draw_landmarks(self,img,draw = True):
        flip_img = cv2.flip(img,1)

        clr_img = cv2.cvtColor(flip_img,cv2.COLOR_BGR2RGB) 

        self.result= self.hand_landmarks.process(clr_img)

        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(flip_img,hand,self.mp_hands.HAND_CONNECTIONS)


        return flip_img
    
    def get_landmark(self,img,handno = 0,draw = False):
        landmarks = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[handno]
            for id , lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx,cy = int(lm.x*w) ,int(lm.y*h)
                landmarks.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),15,(0,255,255),cv2.FILLED)

        return landmarks
                    




def main():
    cam = cv2.VideoCapture(0)
    hand = HandRecognition()
    ptime = 0
    ctime = 0
    
    while True:
        
        success , img = cam.read()
        flip_img = hand.draw_landmarks(img,draw=True)
        landmarks = hand.get_landmark(flip_img,draw=False)
        
        if not success:
            break

        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime

        cv2.putText(flip_img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),4)
        cv2.imshow("Video Showing",flip_img)

        if len(landmarks) !=0:
            print(landmarks[4])

        if cv2.waitKey(1) & 0xFF == ord("q"):  
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()