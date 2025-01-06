import cv2
import mediapipe as mp
import numpy as np
import collections
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define gesture recognizer
GESTURES = ["Static", "Waving"]

# Buffer to store a sequence of landmarks (e.g., last 30 frames)
frame_buffer = collections.deque(maxlen=30)

# Function to analyze motion

def analyze_motion(buffer):
    """Analyze motion patterns over a sequence of frames."""
    motion_features = []

    for i in range(1, len(buffer)):
        # Compute velocity for each landmark
        prev_frame = np.array(buffer[i - 1])
        curr_frame = np.array(buffer[i])
        velocity = curr_frame - prev_frame
        motion_features.append(velocity)

    # Example: Recognize a motion pattern (e.g., 'wave')
    motion_features = np.array(motion_features)
    avg_motion = np.mean(motion_features, axis=0)  # Average motion over the sequence
    if np.linalg.norm(avg_motion) > 0.1:  # Example threshold
        return "Waving"
    return "Static"

# Build a simple LSTM model for demonstration purposes
def build_model():
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=(45, 42)),  # 21 landmarks * 2 (x, y) for each hand
        Dense(64, activation='relu'),
        Dense(len(GESTURES), activation='softmax')  # Number of gestures
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load or initialize a trained model
model = build_model()

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Starting ASL Interpreter with Motion Detection. Press 'q' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert the image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(rgb_frame)

    # Draw hand landmarks and analyze motion
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Store landmarks in the buffer
            landmarks = [
                (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
            ]
            frame_buffer.append(landmarks)

    # Ensure buffer has enough data before analyzing
    if len(frame_buffer) == frame_buffer.maxlen:
        sequence = np.array(frame_buffer).reshape(1, -1, 42)  # Adjust for model input shape
        motion_pattern = analyze_motion(frame_buffer)
        prediction = model.predict(sequence)
        gesture = GESTURES[np.argmax(prediction)]

        # Display the motion pattern or gesture
        cv2.putText(frame, f"Motion: {motion_pattern}, Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('ASL Interpreter with Motion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()