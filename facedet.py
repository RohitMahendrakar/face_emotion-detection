import cv2
from fer import FER
#import time

detector=FER(mtcnn=True)

img = cv2.imread(r'C:\SDP\happy_baby.jpg')
cv2.imshow('frames',img)
print(detector.detect_emotions(img))
emotion,score=detector.top_emotion(img)
print(emotion,score)

##cv2.destroyAllWindows()
'''for video capture'''
'''
while True:
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    print(detector.detect_emotions(frame))
    cap.release();
    cv2.imshow("image from camera",frame)
    if cv2.waitKey(1)&0xFF == ord('q'):
        break
cv2.videoCapture.release()
cv2.destroyAllWindows()
'''
