import numpy as np
import os
from config import DATA_PATH, ACTIONS

def preprocess_data(fixed_sequence_length=30):  # Set a default sequence length, e.g., 30
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(ACTIONS)}

    for file in os.listdir(DATA_PATH):
        for sequence in os.listdir(os.path.join(DATA_PATH, file)):
            frame_dir = os.path.join(DATA_PATH, file, sequence)
            frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.npy')])
            window = []

            for frame_num in range(min(fixed_sequence_length, len(frame_files))):
                res = np.load(os.path.join(frame_dir, f"{frame_num}.npy"))
                window.append(res)
            
            # Pad with zeros if the sequence is shorter than fixed_sequence_length
            while len(window) < fixed_sequence_length:
                window.append(np.zeros_like(window[0]))  # Pad with a zero array of the same shape as the frames

            sequences.append(window)
            labels.append(label_map[file])

    return np.array(sequences), np.array(labels)
