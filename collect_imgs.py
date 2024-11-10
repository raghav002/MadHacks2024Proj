# Library providing ways to interact with the operating system
import os
# Open Source Computer Vision Library -> Used for real-time computer vision and image processing tasks 
import cv2


# Path specification for the folder where all the data of the images will be stored 
DATA_DIR = './data'
# If statement saying that if this path does not exist, make it 
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Number of classes here is basically number of letters. As of now, we will train the images for only 3 letters
number_of_classes = 19
# The number of images to be taken for each of the letters. Higher is usually better, alongside more diverse + 
# awkward positions 
dataset_size = 100

# Function used to capture video from computer's camera. Can also be used with a video file or an IP camera/network stream
cap = cv2.VideoCapture(0)  
# Basically saying that if the camera isn't functioning print an error and exit 
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through each class, in this case each letter 
for j in range(number_of_classes):
    #This makes a subfolder for each of the classes, starting with 0. This means if we start with letters, a will be file 0
    #and z will be file 25 in data. Numbers will come after that, as will words
    class_dir = os.path.join(DATA_DIR, str(j))
    # Same thing as earlier - make the directory if it doesn't already exist 
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    # Formatted string literal printing a message to console where variable j is directly embedded in the string
    # j will be converted to the value it represents, here a numerical value 
    print(f'Collecting data for class {j}')

    # Wait for the user to be ready to start capturing images. So set default of 'done' to False
    done = False
    # While done is not True, do this (will exit loop when done becomes True)
    while not done:
        # Ret = boolean value indicating if frame was captured, frame is a numpy array repsenting captured frame
        # frame is the image in BGR format that can be further processed / displayed
        # This function captures a frame from a video stream
        ret, frame = cap.read()
        # If it isn't captured properly, or the frame is somehow null, throw this error message
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break
        
        # Function used to draw text on an image or video frame. Here, the frame is the frame we capture earlier
        # Format is cv2.putText(image, text, position, font, font_scale, color, thickness, line_type (LINE_AA here), bottomLeftOrigin)
        cv2.putText(frame, 'Ready? Press "Q" to start capturing images.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        # Function used to display an image or video frame in a window. Typically used for debugging/visualizing results of 
        # image processing -> showing input image, output after transformations, intermediate steps during cv task
        # brackets -> (window_name, image)
        cv2.imshow('frame', frame)
        # Check if the user has pressed the Q key. 25 in this context is the time in milliseconds it'll wait for a key press
        # return value is an integer representing the ASCII value of the key pressed -> If none pressed in time, returns -1
        # ord() -> Returns unicode code point/integer rep of character passed -> returns the ASCII value of specificed value
        # here of 'q' -> 113
        if cv2.waitKey(25) == ord('q'): 
            # Change done to true to indicate the user has begun capture 
            done = True

    # Capture the specified number of images for the current class
    counter = 0
    # While 100 images have not been reached
    while counter < dataset_size:
        # Capture frame
        ret, frame = cap.read()
        # Error message as shown earlier
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break
        # Show frame as specified earlier
        cv2.imshow('frame', frame)
        # Wait 25 milliseconds before next capture
        cv2.waitKey(25)
        # Function is used to save an image file to the disk. Allows you to specify file path and format in which the 
        # image should be saved 
        # Brackets -> (filename, image)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

    print(f'Collected {dataset_size} images for class {j}')

# Function to release resources associated with cv2.VideoCapture object (in this case, the camera)
# Prevenets resource leaks 
cap.release()
# Function used to close all windows opened by the cv2.imshow() function -> windows remain open unless explicitly
# told 
cv2.destroyAllWindows()
