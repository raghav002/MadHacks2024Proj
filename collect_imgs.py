import os
import cv2

<<<<<<< Updated upstream
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Try using the default camera index 0 for the MacBook's built-in camera
cap = cv2.VideoCapture(0)  # Use 0 for MacBook's camera
=======
# Path specification for the folder where all the data of the images will be stored 
DATA_DIR = './data'

# If the path does not exist, make it 
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes here is basically the number of letters. In this case, 26 letters (A to Z)
number_of_classes = 26
# The number of images to be taken for each of the letters. Higher is usually better, alongside more diverse + awkward positions
dataset_size = 100

# Function used to capture video from computer's camera.
cap = cv2.VideoCapture(0)  
# If the camera isn't functioning, print an error and exit 
>>>>>>> Stashed changes
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

<<<<<<< Updated upstream
# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')
=======
# Loop through each class, in this case, each letter from 'A' to 'Z'
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, chr(j + 65))  # Using 'A' to 'Z' (65 = ASCII value of 'A')
    
    # If directory doesn't exist, skip it
    if not os.path.exists(class_dir):
        print(f"Directory {chr(j + 65)} not found, skipping.")
        continue

    # If the directory for the class doesn't exist, create it 
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    print(f'Collecting data for class {chr(j + 65)}')
>>>>>>> Stashed changes

    # Wait for the user to be ready to start capturing images
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break
<<<<<<< Updated upstream

        cv2.putText(frame, 'Ready? Press "Q" to start capturing images.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):  # Press "Q" to start capturing images
=======
        
        # Display a message to prompt user
        cv2.putText(frame, 'Ready? Press "Q" to start capturing images.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Show the current frame
        cv2.imshow('frame', frame)
        
        # Wait for the user to press 'Q' to start capturing
        if cv2.waitKey(25) == ord('q'):
>>>>>>> Stashed changes
            done = True

    # Capture the specified number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break
<<<<<<< Updated upstream

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
=======
        
        # Show the captured frame
        cv2.imshow('frame', frame)
        
        # Wait 25 milliseconds before capturing the next frame
        cv2.waitKey(25)
        
        # Save the captured frame to the appropriate directory
>>>>>>> Stashed changes
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

    print(f'Collected {dataset_size} images for class {chr(j + 65)}')

<<<<<<< Updated upstream
=======
# Release the camera and close all windows
>>>>>>> Stashed changes
cap.release()
cv2.destroyAllWindows()
