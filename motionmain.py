from tensorflow import load_model   
import cv2
from playsound import playsound   
import numpy as np
import mediapipe as mp 
import streamlit as st  
from motionutils import mediapipe_detection, draw_styled_landmarks, prob_viz, extract_keypoints, mp_holistic
from config import THRESHOLD, sentence, ACTIONS
import numpy as np
import matplotlib.pyplot as plt 

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
            predicted_action = "" 
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_index = np.argmax(res)
                predicted_action = ACTIONS[action_index]  # Get the action label
                predictions.append(action_index)
                #res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(ACTIONS[np.argmax(res)])
                #predictions.append(np.argmax(res))

                #3. Viz logic
                if np.unique(predictions[-10:])[0] == action_index:
                            if res[action_index] > threshold:
                                if len(sentence) > 0:
                                    if predicted_action != sentence[-1]:
                                        sentence.append(predicted_action)
                                else:
                                    sentence.append(predicted_action)
                                    
            # Display the predicted action on the frame
            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)  # Background rectangle
            cv2.putText(
                image, f"Action: {predicted_action}", (10, 30),  # Display action text
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
            )

            cv2.imshow('Action Recognition', image)

            # Show the frame in Streamlit
            FRAME_WINDOW.image(image, channels="BGR")  # Display the processed frame

            # Break gracefully if 'q' is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

main()



