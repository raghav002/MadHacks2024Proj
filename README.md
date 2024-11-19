# MadHacks2024Proj
MadHacks 2024 Project Repository - Duolingo for Sign Language


# Hand Gesture Recognition and Classification

This project is a hand gesture recognition and classification system using machine learning and computer vision techniques. The system is built using Python, Flask, and various machine learning libraries.

## Introduction

This project enables users to practice sign language by recognizing hand gestures through their webcam. The system classifies the gestures and provides real-time feedback. It is implemented using **Python**, **Flask**, and various machine learning libraries, with deep learning models for hand gesture recognition.

The project includes a user-friendly web application built with **Flask**, which provides a dynamic user interface for practicing sign language. The web app integrates hand gesture recognition models that have been trained on a collection of sign language images.

Key features of this project:
- **Hand Gesture Recognition**: Using webcam input, the system recognizes and classifies gestures for letters, numbers, and words in sign language.
- **Interactive Interface**: Users can practice signing letters, numbers, and words through the application.
- **Quizzes**: Includes multiple quiz options for reinforcing sign language learning.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Endpoints](#endpoints)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/hand-gesture-recognition.git
    cd hand-gesture-recognition
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Collect images for training the model (see [Data Collection](#data-collection)).
2. Train the model using the collected data (see [Training the Model](#training-the-model)).
3. Run the Flask application (see [Running the Application](#running-the-application)).

## Project Structure

```
.DS_Store
app.py
collect_imgs.py
create_dataset.py
data/
    0/
    1/
    10/
    11/
    12/
    13/
    14/
    15/
    16/
    17/
    18/
    2/
    3/
    4/
    5/
    6/
    7/
    8/
    9/
data.pickle
inference_classifier.py
License
model.p
README.md
requirements.txt
templates/
    .DS_Store
    letters.html
    mainmenu.html
    numbers.html
    practicemenu.html
    quizpage.html
    resources.html
    results.html
    words.html
train_classifier.py
```

## Data Collection

To collect images for training the model, run the `collect_imgs.py` script. This script will capture images from your webcam and save them in the `data/` directory.

```sh
python collect_imgs.py
```

## Training the Model

To train the model using the collected data, run the `train_classifier.py` script. This script will train a machine learning model and save it as `model.p`.

```sh
python train_classifier.py
```

## Running the Application

To run the Flask application, execute the `app.py` script. This will start a local web server where you can interact with the hand gesture recognition system.

```sh
python app.py
```

## Endpoints

- `/` - Main menu
- `/practicemenu` - Practice menu
- `/resources` - Resource menu
- `/letters` - View letter representations
- `/numbers` - View number representations
- `/words` - View words
- `/quizpage` - Quiz page
- `/quizpage2` - Quiz with just letters
- `/quizpage3` - Quiz with just numbers
- `/quizpage4` - Quiz with just words
- `/quizpage5` - Quiz with hybrid
- `/results/<int:score>/<int:total>` - Results page
- `/predict_gesture` - Endpoint for predicting hand gestures
