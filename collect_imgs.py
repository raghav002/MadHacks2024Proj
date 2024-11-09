import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Try using the default camera index 0 for the MacBook's built-in camera
cap = cv2.VideoCapture(0)  # Use 0 for MacBook's camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Loop through each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for the user to be ready to start capturing images
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break

        cv2.putText(frame, 'Ready? Press "Q" to start capturing images.', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if cv2.waitKey(25) == ord('q'):  # Press "Q" to start capturing images
            done = True

    # Capture the specified number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Failed to capture image. Check camera connection.")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)

        counter += 1

    print(f'Collected {dataset_size} images for class {j}')

cap.release()
cv2.destroyAllWindows()
