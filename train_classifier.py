# Data serializer/deserializer (for Pyth objects)
import pickle
# Sklearn = ML algo and tool for model selections/pre-processing/evaluation
# ensemble = method that constructs a set of classifiers and then classify new 
# data points by taking a vote
# this module includes such methods -> Create multiple models and combine to get 
# imrpoved results. Other exampels include GradientBoosting, AdaBoost, etc.
# RandomForestClassifier = ensemble learning method for classification
# it combines multiple decision trees to improve the accuracy of the model
from sklearn.ensemble import RandomForestClassifier
# provides tools for model selection and eval, including splitting data into training and
# testing sets whichi is what you see here, cross-validation, and parameter tuning
from sklearn.model_selection import train_test_split
# Provides functions for eval the performance of ML models. Gives metrics for
# classification, regression, and clustering algorithms
from sklearn.metrics import accuracy_score
# Package for scientific computing -> A lot of matrice math/array stuff/math functions
import numpy as np
import os
import cv2
# Solutions for sensory perception related tasks
import mediapipe as mp

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))
# Convert dictionary to numpy arrays, in both cases 
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model
model = RandomForestClassifier()

# Train the model
model.fit(x_train, y_train)

# Predict on test data
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
