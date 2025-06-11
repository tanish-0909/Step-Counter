#version 1.0.0
import cv2
import fontTools.afmLib
from cvzone.PoseModule import PoseDetector

cap = cv2.VideoCapture(1)

detector = PoseDetector()
position_list = []

while True:
    success, img = cap.read()
    img = detector.findPose(img)

    lm_list, b_box_info = detector.findPosition(img)
    # if b_box_info:  #if it isnt empty, i.e. a body is detected
    #     lm_str = ''
    #     for i in lm_list:
    #         #i is a list of 4 integers of the form [index, x, y, z]
    #         lm_str = lm_str + f'{i[1]}, {img.shape[0] - i[2]}, {i[3]},'      #creating csv
    #         #for the y coordinate, cv2 uses y coord from top to bottom, unity uses it bottom to top, so changing that.
    #     position_list.append(lm_str)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        with open('landmark_posn.txt', 'w') as f :
            f.writelines(["%s\n" % item for item in position_list])
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()

