from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import pickle
import mediapipe as mp

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: '1',
               10: '2', 11: '3', 12: '4', 13: '5', 14: 'Hang loose', 15: 'Dislike', 16: 'Shocker', 17: 'Good luck', 
               18: 'Rock on'}

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route('/')
def main_menu():
    return render_template('mainmenu.html')

@app.route('/practicemenu')
def practice_menu():
    return render_template('practicemenu.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/letters')
def letters():
    return render_template('letters.html')

@app.route('/numbers')
def numbers():
    return render_template('numbers.html')

@app.route('/words')
def words():
    return render_template('words.html')

@app.route('/quizpage')
def quiz_page():
    quiz_words = list(labels_dict.values())  # Load all possible words for the quiz
    return render_template('quizpage.html', quiz_words=quiz_words)

# Quiz with just letters
@app.route('/quizpage2')
def quiz_page2(): 
    quiz_words2 = list(labels_dict.values())
    letters= quiz_words2[:9]
    return render_template('quizpage.html', quiz_words=letters)

# Quiz with just numbers
@app.route('/quizpage3')
def quiz_page3(): 
    quiz_words3 = list(labels_dict.values())
    numbers = quiz_words3[9:14]
    return render_template('quizpage.html', quiz_words=numbers)

# Quiz with just words
@app.route('/quizpage4')
def quiz_page4(): 
    quiz_words4 = list(labels_dict.values())
    words = quiz_words4[14:19]
    return render_template('quizpage.html', quiz_words=words)

# Quiz with hybrid
@app.route('/quizpage5')
def quiz_page5(): 
    quiz_words5 = list(labels_dict.values())
    randomeles = [quiz_words5[i] for i in [2, 3, 10, 11, 16, 17]]
    return render_template('quizpage.html', quiz_words=randomeles)

@app.route('/results/<int:score>/<int:total>')
def results(score, total):
    return render_template('results.html', score=score, total=total)

@app.route('/predict_gesture', methods=['POST'])
def predict_gesture():
    file = request.files['video_frame'].read()
    np_img = np.frombuffer(file, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        data_aux = []
        x_ = []
        y_ = []

        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        for landmark in hand_landmarks.landmark:
            data_aux.append(landmark.x - min(x_))
            data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Calculate bounding box coordinates
        H, W, _ = frame.shape
        x1, y1 = int(min(x_) * W), int(min(y_) * H)
        x2, y2 = int(max(x_) * W), int(max(y_) * H)

        return jsonify({
            "prediction": predicted_character,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

    return jsonify({"prediction": None})

if __name__ == '__main__':
    app.run(debug=True)
