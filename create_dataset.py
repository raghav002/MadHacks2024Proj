import os
# Library used to serialize/deserialize Python objects -> convering Pyth object into 
# byte stream which can be saved to a file or transmitted over a network
# deser -> reverse 
import pickle
# Cross-platform framework deved by Google for creating multisensing machine learning pipelines
# video/audio/sensor. Has pre-built ML solns for face detection, hand tracking, pose detection, etc.
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands module
# Provides hand tracking solution
mp_hands = mp.solutions.hands
# Provides utilities for drawing landmarks and connections -> The skeletal outline 
mp_drawing = mp.solutions.drawing_utils
# Provides drawing styles for those landmarks and connections
mp_drawing_styles = mp.solutions.drawing_styles

# Class provided for hand tracking -> static image mode being true means it'll treat the 
# input images as static images, detecting hands in every frame. If False, treats input as 
# video stream and tracks hands across frames
# Min detect confidence = min confidence value (between 0 and 1) for hand detection to be 
# considered successful. A higher value means means algo will be more confident that it detects
# hands 
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Remember this from collect_imgs
DATA_DIR = './data'
data = []
labels = []
# For each of the directories in the data directory
for dir_ in os.listdir(DATA_DIR):
    #Let the directory path be this directory + the current value dir_ -> remember that 
    #the folders for a-z (in this case) are labelled 0-25
    dir_path = os.path.join(DATA_DIR, dir_)
    # If the path is a directory
    if os.path.isdir(dir_path):
        # For each image in the directory
        for img_path in os.listdir(dir_path):
            # Will store normalized landmark values
            data_aux = []
            # Will store x values of landmarks
            x_ = []
            # Will store y values of landmarks 
            y_ = []

            # Read image using imread 
            img = cv2.imread(os.path.join(dir_path, img_path))
            # Function to convert color space to another. In this case, converts an image
            # from BGR to RGB (each letter is a color)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to extract hand landmarks
            results = hands.process(img_rgb)

            # If hand landmarks are detected
            if results.multi_hand_landmarks:
                # For each hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks for the hand by iterating over each landmark
                    for i in range(len(hand_landmarks.landmark)):
                        # Get the x and y coordinates of the landmark. It has z as well, 
                        # but we are not using it in this case
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    # Normalize the coordinates and add to data
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        # Normalization -> subtracting the minimum value from the list of x 
                        # and y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append the data and labels
                # Append landmark data to data list
                data.append(data_aux)
                # Append which letter this is for to the labels list
                labels.append(dir_)

# Save the dataset to a pickle file -> contains serialized Python objects 
f = open('data.pickle', 'wb')
# This serializes the data and labels and writes them to the file
pickle.dump({'data': data, 'labels': labels}, f)
# Close the file
f.close()
