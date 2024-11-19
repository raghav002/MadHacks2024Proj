import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)  

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Initialize MediaPipe hands solution
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: '1',
               10: '2', 11: '3', 12: '4', 13: '5', 14: 'Hang loose', 15: 'Dislike', 16: 'Shocker', 17: 'Good luck', 
               18: 'Rock on'}

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture image. Exiting...")
        break
    #Unpacking dimensions of frame into three variables - H, W, and _ 
    # H = Height, W = width, _ = number of channels (3 for an RGB image)
    # Underscore is used as a placeholder for a variable that we don't need
    H, W, _ = frame.shape

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(frame_rgb)
    
    data_aux = []
    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        # Process only the first hand (if only single hand is needed)
        # If two hands -> results.multi_hand_landmarks[:2]
        # If no limit -> results.multi_hand_landmarks
        for hand_landmarks in results.multi_hand_landmarks:  
            x_ = []
            y_ = []

            # Extract hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # Normalize and add to data list
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            # Draw bounding box -> Shows the area where the hand is detected
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Prediction
            # Method from the ML model (a classifier) that makes a prediction based
            # on the input data (in this case, the normalized hand landmarks)
            # Prediction is the output of the model -> the predicted class/letter
            prediction = model.predict([np.asarray(data_aux)])
            # labels_dict is a dict that maps class indices to their corresponding labels/chars
            # the stuff in brackets extracts the predicted class index from the prediction array
            # The whole thing retrieves the corresponding character/letter for the predicted class
            # and stores it in the variable predicted_character
            # The variable itself is the char or label that the model predicts based on the 
            # input data
            predicted_character = labels_dict[int(prediction[0])]

            # Display bounding box and predicted character to show what the model things you're
            # signing
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)

    # Show the frame with predicted gesture
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
