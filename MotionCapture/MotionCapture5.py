import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize step counting variables.
previous_left_ankle_y = None
previous_right_ankle_y = None
left_step_count = 0
right_step_count = 0
step_threshold = 20  # Adjust this threshold based on your camera setup and person's height.

# Initialize the video capture from the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the pose annotations on the image.
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract leg landmarks.
        landmarks = results.pose_landmarks.landmark
        left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y * frame.shape[0]
        right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame.shape[0]

        # Detect steps based on ankle movement.
        if previous_left_ankle_y is not None and previous_right_ankle_y is not None:
            if abs(left_ankle_y - previous_left_ankle_y) > step_threshold:
                left_step_count += 1
                previous_left_ankle_y = left_ankle_y
            if abs(right_ankle_y - previous_right_ankle_y) > step_threshold:
                right_step_count += 1
                previous_right_ankle_y = right_ankle_y
        else:
            previous_left_ankle_y = left_ankle_y
            previous_right_ankle_y = right_ankle_y

        # Display step count.
        cv2.putText(image, f"Left Steps: {left_step_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f"Right Steps: {right_step_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the image.
    cv2.imshow('Real-Time Step Counting', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources.
pose.close()
cap.release()
cv2.destroyAllWindows()
