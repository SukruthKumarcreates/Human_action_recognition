import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load model and labels
with open("har_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("action_labels.txt", "r") as f:
    action_labels = [line.strip() for line in f.readlines()]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

        if len(landmarks) == len(model.feature_importances_):  # Ensure feature size match
            input_data = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data).max()

            cv2.putText(frame, f"{prediction} ({probability:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("HAR - Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
