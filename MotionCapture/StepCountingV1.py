import cv2
import numpy as np
from math import sqrt
from cvzone.PoseModule import PoseDetector
import matplotlib.pyplot as plt
from cv2 import imshow

cap = cv2.VideoCapture(0)

detector = PoseDetector()

def distance(a, b):
    return(sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2))

def angle(a, b, c):
    ca = a - c
    bc = b - c
    cosine_angle = np.dot(ca, bc) / (np.linalg.norm(ca) * np.linalg.norm(bc))
    return cosine_angle


def initialize_threshold(cap, detector, num_frames=60):
    li2 = []
    point_1 = 0  # Initial point_1 value

    for _ in range(num_frames):
        success, img = cap.read()
        if not success:
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)

        if bboxInfo:
            lm_left_ankle = lmList[28]
            lm_right_ankle = lmList[27]

            # if not(lm_right_ankle and lm_right_ankle):
            #     print("legs not found")
            #     continue
            #
            # Calculate distance between ankles
            point_2 = distance(lm_right_ankle, lm_left_ankle)

            # Apply exponential filter
            point_1 = 0.5 * point_1 + 0.5 * point_2
            li2.append(point_1)
            #imshow("configuring", img)
        elif not bboxInfo:
            j = j-1;
            print("Human not detected.")
            continue;

    if li2:
        maxi = max(li2)
        mini = min(li2)
        threshold = (maxi + mini) / 2
        return threshold, li2[num_frames - 1]
    else:
        raise ValueError("Error: Video not found or unable to read frames.")

def main_processing_loop(cap, detector, threshold, steps=2, time=60, peak_loc=0, point_1 = 0.0):
    # cv2.destroyAllWindows()
    maxi = 0
    mini = 0

    #li = []
    while True:

        success, img = cap.read()
        if not success:
            print("Video ended.")
            break

        img = detector.findPose(img)
        lmList, bboxInfo = detector.findPosition(img)

        time += 1

        if bboxInfo:
            lm_right_hip = lmList[23]
            lm_left_hip = lmList[24]
            lm_right_knee = lmList[25]
            lm_left_knee = lmList[26]
            lm_left_ankle = lmList[28]
            lm_right_ankle = lmList[27]

            if not(lm_right_ankle and lm_right_ankle):  #to be fixed
                print("legs not found")
                continue

            # Calculate distance between ankles
            point_2 = distance(lm_right_ankle, lm_left_ankle)

            # Apply exponential filter
            point_1_exp = 0.5 * point_1 + 0.5 * point_2
            #li.append(point_1_exp)

            if point_1_exp > threshold and point_1 < threshold:
                if time - peak_loc >= 12:  # 12 can be adjusted based on requirements
                    steps += 1
                    peak_loc = time
                    threshold = 0.25 * (maxi + mini) + 0.5 * threshold
                    #print(threshold)
                    maxi = point_1
                    mini = point_1

            if point_1_exp > maxi:
                maxi = point_1_exp
            if point_1_exp < mini:
                mini = point_1_exp

            point_1 = point_1_exp
        elif not bboxInfo:
            print("Human not detected.")
            continue;

        # Display the image
        imshow("video", img)

        # Break loop on ESC key press
        key = cv2.waitKey(1)
        if key == 27:
            break


    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return steps

#main()

# Initialize threshold
try:
    threshold, point1 = initialize_threshold(cap, detector)
except ValueError as e:
    print(e)
else:
    # Main processing loop
    print(steps := main_processing_loop(cap, detector, threshold, point_1 = point1))

#
cap.release()
cv2.destroyAllWindows()

# print(steps)
#plt.plot(li2+li)
# print(threshold)
#plt.show()