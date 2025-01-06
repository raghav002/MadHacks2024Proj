from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
from data_collection import collect_data
from preprocessing import preprocess_data
import os
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
import json
from config import ACTIONS
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

def create_model():
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(30,1662)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(len(ACTIONS), activation='softmax'))

    return model


def train_model(X_train, y_train):
    model = create_model()
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    log_dir = os.path.join('Logs')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tb_callback = TensorBoard(log_dir='./Logs')
    model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])
    return model

def collect_and_train():
    # with open('final_train.json', 'r') as data_file:
    #     json_data = data_file.read()

    # instance_json = json.loads(json_data)
    # actions = list(instance_json.keys())
    #Data Collection
    # collect_data()   
    # Data Preprocessing
    X, y = preprocess_data()   
    y = to_categorical(y, num_classes=len(ACTIONS))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    # Model Training
    model = create_model()
    model = train_model(X_train, y_train)
    # model = load_model('action.h5') # // To delete after testing
    res = model.predict(X_test)
    print(len(y_test))
    print(np.sum(np.array(ACTIONS)[np.argmax(res,axis =1)] == np.array(ACTIONS)[np.argmax(y_test,axis =1)]))
    print(np.array(ACTIONS)[np.argmax(res,axis =1)])
    print(np.array(ACTIONS)[np.argmax(y_test,axis =1)])
    model.save('action.h5')

if __name__ == "__main__":
    collect_and_train()
