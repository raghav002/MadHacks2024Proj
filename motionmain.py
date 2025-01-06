from tensorflow.keras.models import load_model
import json
from gtts import gTTS
import os
import cv2
from playsound import playsound
import numpy as np
from config import ACTIONS
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import streamlit as st
import time
import cv2
from utils import mediapipe_detection, draw_styled_landmarks, prob_viz, extract_keypoints, mp_holistic
from config import THRESHOLD, sentence
import numpy as np
import matplotlib.pyplot as plt
import copy

# def main():
#     language = 'en'
#     model = load_model('action.h5')
#     cap = cv2.VideoCapture(0)
#     # Real-time Detection
#     #while(True):
#     sentence = detect_in_real_time(model, ACTIONS, cap) #Call this function if start button pressed
#     fixed_text = gptgen(sentence)
#     #myobj = gTTS(text=fixed_text, lang=language, slow=False)
#     print(sentence)
#     print(fixed_text)
#     # myobj.save("text.mp3")
#     # playsound("text.mp3")
#     # if cv2.waitKey(10) & 0xFF == ord('q'): 
#     #     break

#     cap.release()
#     cv2.destroyAllWindows()

# sentence = []
# temp = []
# def prompt():
#     global temp
#     return temp



def main():
    # global temp
    # global sentence
    sequence = []
    
    predictions = []
    threshold = 0.5

    # Load the model once before the loop
    model = load_model('action.h5')

    cap = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])  # Initialize Streamlit image display
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to open the camera.")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
        
            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(ACTIONS[np.argmax(res)])
                predictions.append(np.argmax(res))

        #3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if ACTIONS[np.argmax(res)] != sentence[-1]:
                                sentence.append(ACTIONS[np.argmax(res)])
                        else:
                            sentence.append(ACTIONS[np.argmax(res)])


                # if len(sentence) > 5: 
                #     sentence = sentence[-5:]

            # Draw sentence
            #cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            #cv2.putText(image, ' '.join(sentence), (3, 30), 
             #           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
            # Show the frame in Streamlit
            FRAME_WINDOW.image(image, channels="BGR")  # Display the processed frame

            # Break gracefully if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()



