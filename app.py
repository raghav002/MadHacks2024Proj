import pickle
import cv2
import mediapipe as mp
import numpy as np
# Lightweight web framework for python -> designed for easy web dev with min overhead
# Response -> Create HTTP response objects -> Customize response Flask app sends to client
# Can set response code, headers, data, etc.
# jsonify -> Converts a dict to a JSON response
# JSON response -> HTTP response containing data formatted as JSON. Lightweight data interchange
# format that is easy for humans to read and write, easy for machines to aprse and gen
# commonly used for transmitting data in web applications 
# request -> Used to access incoming request data. Object containing all data sent by client
# in an HTTP request 
from flask import Flask, render_template, Response, jsonify, request
# Provides functions to work with time related tasks -> getting current time, pausing exec
# for specified time, measuring time taken by code exec, etc. 
import time

app = Flask(__name__)

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Labels for gesture recognition -> Update when you add the rest of the letters 
labels_dict = {0: 'A', 1: 'B', 2: 'L'}
signs = ['A', 'B', 'L']  # For simplicity, use just A, B, L here

# Store current question index and correct answers -> This is for the quiz 
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
    # Remember that signs here is a list with the letters in order (for now) 
    return render_template('quiz.html', sign=signs[current_sign_index])  # Show the current sign

# Function to capture the webcam feed and return it as a video stream
def gen_frames():
    # Remember this activates your camera 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Declares these variables as global variables -> Can be defined at the module level and 
    # can be accessed/modifed by any function in the module
    global sign_start_time, sign_held_for_correct_time


    while True:
        # Read the frames of incoming video data 
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(frame_rgb)
        # Landmark x and y storage
        data_aux = []
        if results.multi_hand_landmarks:
            # CHANGE THIS -> PARTIALLY CAUSING CRASHES ALMOST FOR CERTAIN 
            # Process only the first hand (if only single hand is needed) 
            for hand_landmarks in results.multi_hand_landmarks:
                x_ = []
                y_ = []
                # Store data for each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)
                # Normalize each landmark's data 
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Predict what symbol the user is making 
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                # Check if the user signed correctly and hold for 1 second
                # print(current_sign_index)
                if predicted_character == signs[current_sign_index]:
                    # If the timer hasn't started, start it
                    if not sign_start_time:
                        sign_start_time = time.time()  # Start the timer
                    # If the timer has started, check if it has been held for 1 second
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

        # Encodes the current video frame as a jpeg image 
        # imencode -> OpenCV function that encodes an image into a memory buffer
        # ret = boolean value indicating if coding was successful
        # buffer = memory buffer containing the encoded image
        # memory buffer = contiguous memory block used to store data temp while being
        # moved from one place to another
        ret, buffer = cv2.imencode('.jpg', frame)
        # Converts buffer to bytes object -> Necessary for streaming/transmission over a network
        # /being written to a file
        frame = buffer.tobytes()
        # Sends the encoded frame as part of an http response -> Includes appropriate headers
        # and the encoded image data 
        # \r = carriage return, \n = newline -> Used to separate lines in text files
        # Carriage return = control character in text that instructs cursor to move to the 
        # beginning of the line  -> Together with \n create new line in text files and streams
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    # User to create and return an HTTP response that streams video frames to the client
    # Response = Flask class that represents an HTTP response
    # gen_frames = function that generates video frames
    # mimetype = type of data in the response -> MME type used for streaming content where each 
    # part of the multipart content is an image/jpeg and can replace the previous part
    # boundary = frame specifies the boundary string that separates each part
    # of the multiplart message -> Used to delineate diff parts/frames of the stream 
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return render_template('result.html')

# Quiz-related routes
@app.route('/check_prediction', methods=['POST'])
def check_prediction():
    # Global variables declared to allow for modification in modules used in 
    global sign_held_for_correct_time, current_sign_index
    # If sign is held for correct time
    if sign_held_for_correct_time:
        # Mark the current question as correct by resetting the timer + setting the value 
        # to False
        sign_held_for_correct_time = False
        current_sign_index += 1  # Move to the next sign
        
        # Check if we have reached the end of the quiz
        if current_sign_index == len(signs):
            # If quiz is completed, redirect to practice menu
            # This lin creates and returns a JSON response to client -> Remember that 
            # JSON is a lightweight data interchange format that is easy for humans to read
            # jsonify also sets appropriate content type for HTTP response
            # Dict used here indicates operation status, message, and redirect URL
            current_sign_index = 0
            return jsonify({"status": "completed", "message": "You have completed the quiz!", "redirect": "/practice"})
        else:
            return jsonify({"status": "correct", "sign": signs[current_sign_index]})
    # If sign isn't held for enough time or is incorrect
    return jsonify({"status": "incorrect", "message": "Try again."})

if __name__ == '__main__':
    app.run(debug=True)
