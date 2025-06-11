import cv2
import numpy as np
from math import sqrt
from cvzone.PoseModule import PoseDetector
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('videos/Video5.mp4')

detector = PoseDetector()
li = []
point_1 = 0.0
def distance(a, b):
    return(sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))
def angle(a, b, c):
    ca = a - c
    bc = b - c

    cosine_angle = np.dot(ca, bc) / (np.linalg.norm(ca) * np.linalg.norm(bc))
    return cosine_angle

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList, bboxInfo = detector.findPosition(img)

    #print(lmList)
    lm_right_hip = lmList[23]
    lm_left_hip = lmList[24]
    lm_right_knee = lmList[25]
    lm_left_knee = lmList[26]
    lm_left_ankle = lmList[28]
    lm_right_ankle = lmList[27]

#finding angle between them
    # point_2 = angle(np.array(lm_right_knee), np.array(lm_left_knee), (np.array(lm_right_hip) + np.array(lm_left_hip)) / 2)

#finding distance between them
    point_2 = distance(lm_right_ankle, lm_left_ankle)


#exponential filter
    point_1 = 0.5 * point_1 + 0.5 * point_2
    li.append(point_1)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
print(li)
plt.plot(li)

plt.show()