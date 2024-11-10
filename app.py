from flask import Flask, render_template, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Example, add more letters as needed

@app.route('/')
def index():
    # Pass words or letters for the quiz
    quiz_words = list(labels_dict.values())  # Modify as needed
    return render_template('quizpage.html', quiz_words=quiz_words)

@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    # Get the video frame from the client
    file = request.files['video_frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process the frame with MediaPipe to get hand landmarks
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = []
            y_ = []

            # Extract landmarks for the hand
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize landmarks and prepare data for prediction
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Make prediction with the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            return jsonify({'prediction': predicted_character})

    return jsonify({'prediction': None})

if __name__ == '__main__':
    app.run(debug=True)
