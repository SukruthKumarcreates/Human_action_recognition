import cv2
import mediapipe as mp
import pandas as pd
import os

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

label = input("Enter action label (e.g., walking, jumping): ")
print(f"Recording pose data for action: {label}")

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
        data.append(landmarks)
        labels.append(label)

        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Recording - Press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save to CSV
df = pd.DataFrame(data)
df['label'] = labels

if not os.path.exists('dataset'):
    os.mkdir('dataset')

df.to_csv(f'dataset/{label}.csv', index=False)
print(f"Saved {len(data)} records to dataset/{label}.csv")
