import os
import pickle
import mediapipe as mp
import cv2

<<<<<<< Updated upstream
# Initialize MediaPipe Hands module
=======
# Initialize MediaPipe Hands module for hand tracking
>>>>>>> Stashed changes
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

<<<<<<< Updated upstream
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

=======
# Set up the hand tracking solution
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Path for the dataset
>>>>>>> Stashed changes
DATA_DIR = './data'
data = []
labels = []

<<<<<<< Updated upstream
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if os.path.isdir(dir_path):
        for img_path in os.listdir(dir_path):
            data_aux = []
            x_ = []
            y_ = []

            # Read image
            img = cv2.imread(os.path.join(dir_path, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
=======
# Loop through the directories in the data folder (for each letter)
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Check if it's a directory (skip non-directory files)
    if not os.path.isdir(dir_path):
        continue
    
    # Skip if the directory is not a valid class folder (A to Z)
    if dir_ not in [chr(i + 65) for i in range(26)]:  # Check if dir_ is one of the alphabet letters
        print(f"Skipping {dir_} as it's not a valid class folder.")
        continue
    
    # For each image in the directory
    for img_path in os.listdir(dir_path):
        data_aux = []
        x_ = []
        y_ = []

        # Read the image using OpenCV
        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
>>>>>>> Stashed changes

        # Process the image to extract hand landmarks
        results = hands.process(img_rgb)

<<<<<<< Updated upstream
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract landmarks for the hand
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
=======
        # If hand landmarks are detected, process them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)
>>>>>>> Stashed changes

                # Normalize the coordinates and add to the data list
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

<<<<<<< Updated upstream
                    # Normalize the coordinates and add to data
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Append the data and labels
                data.append(data_aux)
                labels.append(dir_)

# Save the dataset to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
=======
            # Append the data and corresponding labels to the lists
            data.append(data_aux)
            labels.append(dir_)

# Save the dataset to a pickle file
with open('data.pickle', 'wb') as file:
    pickle.dump(data, file)
    pickle.dump(labels, file)

# Close the hand tracking solution
hands.close()
>>>>>>> Stashed changes
