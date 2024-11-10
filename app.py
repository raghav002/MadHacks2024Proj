import pickle
import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
import time

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for gesture recognition
labels_dict = {0: 'A', 1: 'B', 2: 'L'}
signs = ['A', 'B', 'L']  # For simplicity, use just A, B, L here

# Store current question index and correct answers
current_sign_index = 0
sign_start_time = None
sign_held_for_correct_time = False

# Route for main menu
@app.route('/')
def main_menu():
    return render_template('main_menu.html')

# Route for practice menu
@app.route('/practice')
def practice_menu():
    return render_template('practice_menu.html')

# Route for quiz page
@app.route('/quiz')
def quiz_page():
    return render_template('quiz.html', sign=signs[current_sign_index])  # Show the current sign

# Function to capture the webcam feed and return it as a video stream
def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    global sign_start_time, sign_held_for_correct_time

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(frame_rgb)

        data_aux = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks[:1]:
                x_ = []
                y_ = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Prediction
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Check if the user signed correctly and hold for 1 second
                if predicted_character == signs[current_sign_index]:
                    if not sign_start_time:
                        sign_start_time = time.time()  # Start the timer
                    elif time.time() - sign_start_time >= 1:
                        sign_held_for_correct_time = True

                # Draw bounding box and predicted character
                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Quiz-related routes
@app.route('/check_prediction', methods=['POST'])
def check_prediction():
    global sign_held_for_correct_time, current_sign_index

    if sign_held_for_correct_time:
        # Mark the current question as correct
        sign_held_for_correct_time = False
        current_sign_index += 1  # Move to the next sign
        
        # Check if we have reached the end of the quiz
        if current_sign_index >= len(signs):
            return jsonify({"status": "completed", "message": "You have completed the quiz!"})
        else:
            return jsonify({"status": "correct", "sign": signs[current_sign_index]})

    return jsonify({"status": "incorrect", "message": "Try again."})

if __name__ == '__main__':
    app.run(debug=True)
