import cv2
import numpy as np
from config import DATA_PATH, DATA_PATH2, ACTIONS
from utils import mediapipe_detection, draw_styled_landmarks, mp_holistic, extract_keypoints
import os

def collect_data():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for file in os.listdir(DATA_PATH):
            seq_path = os.path.join(DATA_PATH, file)
            for seq in os.listdir(seq_path):

            
                #print(name[0])
                # Define the directory where frames are stored for this action and sequence
                frame_dir = os.path.join(DATA_PATH, file, seq)

                # Get list of frame files and sort them to ensure correct order
                if (len(frame_dir) <= 95.705) and (file in ACTIONS):
                    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
                    for frame_num, frame_file in enumerate(frame_files):
                        # Load the frame image
                        frame_path = os.path.join(frame_dir, frame_file)
                        frame = cv2.imread(frame_path)
                        
                        # Process the frame
                        image, results = mediapipe_detection(frame, holistic)
                        draw_styled_landmarks(image, results)
                        
                        # Extract and save keypoints
                        keypoints = extract_keypoints(results)
                        save_dir = os.path.join(DATA_PATH2, file, seq)
                        os.makedirs(save_dir, exist_ok=True)

                        # Save keypoints
                        np.save(os.path.join(save_dir, str(frame_num)), keypoints)
                    print(file)
                        
                        # Show the frame (optional)
            #             cv2.imshow('OpenCV Feed', image)
            #             if cv2.waitKey(10) & 0xFF == ord('q'):
            #                 break
            # cv2.destroyAllWindows()
                else:
                    continue
