import cv2
import numpy as np
import os
import time
import mediapipe as mp
import shutil


class CreateData():
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic # Holistic model
        self.mp_drawing = mp.solutions.drawing_utils # Drawing utilities
        self.mp_face_mesh = mp.solutions.face_mesh
        # Thirty videos worth of data
        self.no_sequences = 15
        
        # Videos are going to be 30 frames in length
        self.sequence_length = 30
        
        # Folder start
        self.start_folder = 1

        self.DATA_PATH = os.path.join('MP_Data')
        # Actions that we try to detect
        self.actions = np.array(["i","name","help", "drink water", "cold", "today", "please", "thankyou", "sorry", "IloveU"])

    def mediapipe_detection(self,image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        image.flags.writeable = True                   # Image is now writeable 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
        return image, results

    def draw_styled_landmarks(self, image, results):
    # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_face_mesh.FACEMESH_CONTOURS, 
                                 self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                                 self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 ) 
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                 self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 ) 
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                 self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) 
        # Draw right hand connections  
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS, 
                                 self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                                 self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 ) 

    def extract_keypoints(self,results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def create_folders(self):
        # Path for exported data, numpy arrays
        if not os.path.exists(self.DATA_PATH):
            os.mkdir(self.DATA_PATH)
         
        for i in self.actions:
            if not os.path.exists(os.path.join(self.DATA_PATH,i)):
                os.mkdir(os.path.join(self.DATA_PATH,i))

    def generate_data(self):
        cap = cv2.VideoCapture(0)
        self.create_folders()
        # Set mediapipe model 
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            # NEW LOOP
            # Loop through actions
            for action in self.actions:
                # Loop through sequences aka videos
                for sequence in range(self.start_folder, self.start_folder+self.no_sequences):
                    # Loop through video length aka sequence length
                    for frame_num in range(self.sequence_length):
        
                        # Read feed
                        ret, frame = cap.read()
        
                        # Make detections
                        image, results = self.mediapipe_detection(frame, holistic)
        
                        # Draw landmarks
                        self.draw_styled_landmarks(image, results)
                        
                        # NEW Apply wait logic
                        if frame_num == 0: 
                            cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(1000)
                        else: 
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                        
                        # NEW Export keypoints
                        keypoints = self.extract_keypoints(results)
                        npy_path = os.path.join(self.DATA_PATH, action, str(sequence), str(frame_num))
                        if not os.path.exists(os.path.join(self.DATA_PATH, action, str(sequence))):
                            os.mkdir(os.path.join(self.DATA_PATH, action, str(sequence)))
                        np.save(npy_path, keypoints)
        
                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
                            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    obj = CreateData()
    print("Started Creating Data")
    obj.generate_data()  
    print("Data Generated")

    