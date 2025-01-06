import numpy as np
import os
import json

DATA_PATH = os.path.join('MP_Data')
DATA_PATH2 = os.path.join('npy')
NO_SEQUENCES = 30
SEQUENCE_LENGTH = 30
START_FOLDER = 30
THRESHOLD = 0.9
# ACTIONS =  [
#     "i", "you", "he", "we", "them",
#     "want", "need", "have", "go", "come", "like", "see", "do", "make", "give", "take", "help", "know", "think",
#     "person", "friend", "family", "food", "water", "house", "school", "work", "day", "night", "time", "bed",
#     "good", "bad", "big", "small", "happy", "sad", "old", "new", "hot", "cold",
#     "who", "what", "where", "when", "why", "how",
#     "yes", "no", "please", "thank you", "sorry"
# ]
# # ACTIONS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# NUMS = {
#     'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10,
#     'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19,
#     't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26
# }
ACTIONS = ["i","name","help", "drink water", "cold", "today","please", "thankyou", "sorry", "IloveU"]
sentence = []





