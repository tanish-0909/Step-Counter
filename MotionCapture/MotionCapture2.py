import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Capture video from a file or webcam.
cap = cv2.VideoCapture('Video.mp4')  # Replace with 0 to use webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for OpenCV.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the leg keypoints.
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        leg_landmarks = [
            landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE],
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE],
            landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE],
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE],
            landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]
        ]
        for landmark in leg_landmarks:
            cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 5, (0, 255, 0), -1)

    # Display the image.
    cv2.imshow('Leg Keypoints', image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

# Release resources.
pose.close()
cap.release()
cv2.destroyAllWindows()
