import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_face.FaceDetection()

while True:

    succ , frame = cam.read()
    
    if not succ:
        break

    clr_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result = face_detector.process(clr_img)

    if result.detections:
        for id , detection in enumerate(result.detections):
            # print(detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            box = detection.location_data.relative_bounding_box
            h,w,c = frame.shape
            bbox = int(box.xmin*w) , int(box.ymin*h) , int(box.width*w) , int(box.height*h)


            cv2.rectangle(frame,bbox,(0,255,0),4)

            cv2.putText(frame,f"{int(detection.score[0]*100)}%",(bbox[0],bbox[1]-10),cv2.FONT_HERSHEY_PLAIN,2,(255,0,255),2)




    cv2.imshow("Face Detection",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
