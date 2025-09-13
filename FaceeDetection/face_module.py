import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon=minDetectionCon
        self.mp_face = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detector = self.mp_face.FaceDetection(self.minDetectionCon)


    def drawFaces(self,frame,draw=True):
        clr_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        result = self.face_detector.process(clr_img)

        if result.detections:
            for id , detection in enumerate(result.detections):
                # print(detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                box = detection.location_data.relative_bounding_box
                h,w,c = frame.shape
                bbox = int(box.xmin*w) , int(box.ymin*h) , int(box.width*w) , int(box.height*h)

                if draw:
                    self.fancyDraw(frame,bbox)
                    cv2.putText(frame,f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_PLAIN,2,(255,255,0),2)
        return frame
    

    def fancyDraw(self,img,bbox,l=30):
        x,y,w,h = bbox
        x1 , y1 = x + w , y + h
        
        cv2.rectangle(img,bbox,(255,0,255),2)
        # Top left
        cv2.line(img,(x,y),(x+l,y),(255,0,255),8)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),8)
        # Top Right
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),8)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),8)
        # Bottom Left
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),8)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),8)
        # Bottom Right
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),8)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),8)
    
def main():
    cam = cv2.VideoCapture(0)
    detector = FaceDetector()
    ptime = 0

    while True:
        succ , frame = cam.read()
        
        if not succ:
            break

        img = detector.drawFaces(frame)

        ctime = time.time()
        fps = 1 / (ctime-ptime)
        ptime = ctime

        cv2.putText(frame,f"FPS: {int(fps)}",(20,50),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

        cv2.imshow("Face Detection",img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()