from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
labels_dict = {0: 'A', 1: 'B', 2: 'L'}  # Update with all your labels

def process_frame(frame):
    # Convert base64 string to numpy array
    nparr = np.frombuffer(base64.b64decode(frame), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    data_aux = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:1]:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            
            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            return predicted_character

    return None  # No hand detected

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    frame = request.json['frame']
    result = process_frame(frame)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)