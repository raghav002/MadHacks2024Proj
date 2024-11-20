<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white)](https://gunicorn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)](https://opencv.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

</div>

# DuoHando 
The Duolino for Sign Language. This project is a hand gesture recognition and classification system using machine
learning and computer vision techniques. The system is built using Python, Flask, and various machine learning libraries.

- <a target="_blank" href="https://devpost.com/software/sign-language-learning-platform">Madhacks Devpost</a>
- <a target="_blank" href="https://youtu.be/a8s0QXTINiY">Demo video</a>
- <a target="_blank" href="https://docs.google.com/presentation/d/1OAzeMWcmFGh7TqJFrLm0Mw7lxbCbLjc4gu1ygqOcCI4/mobilepresent?slide=id.g3183c35df13_2_75">Demo Presentation</a>

## Introduction
This project enables users to practice sign language by recognizing hand gestures through their webcam. This application primarily focuses on gamifying the process, expanding the learning resources on an under represented place of study. The system classifies the gestures and provides real-time feedback. The tech stack is further described below, however it is written in **Python**<img align="center" height="20" src="https://cdn.simpleicons.org/python"> with **Flask** <img align="center" src="https://www.vectorlogo.zone/logos/palletsprojects_flask/palletsprojects_flask-icon.svg" width="20"> with various machine learning libraries, and deep learning models for hand gesture recognition.

The project includes a user-friendly web application written behind **Flask** <img align="center" src="https://www.vectorlogo.zone/logos/palletsprojects_flask/palletsprojects_flask-icon.svg" width="20">, on top of <img align="center" height="20" src="https://cdn.simpleicons.org/html5">/<img align="center" height="20" src="https://cdn.simpleicons.org/css3">/<img align="center" height="20" src="https://cdn.simpleicons.org/javascript">, which provides a dynamic user interface for practicing sign language. The web app integrates hand gesture recognition models that have been trained on a collection of sign language images.

Key features of this project:
- **Hand Gesture Recognition**: Using webcam input, the system recognizes and classifies gestures for letters, numbers, and words in sign language.
- **Interactive Interface**: Users can practice signing letters, numbers, and words through a vibrant and easy-to-follow interface.
- **Learning Resources**: Users can find learning resources embedded into the interface, enabling a learn and reinforce study model.
- **Quizzes**: Includes multiple quiz options for reinforcing sign language learning.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Data Collection](#data-collection)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Tech Stack](#tech-stack)
- [Production Implementation](#production-implementation)
- [Endpoints](#endpoints)

## Tech Stack

<div align="left">

- **Python** <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20"> Core programming language

- **Flask** <img align="center" src="https://www.vectorlogo.zone/logos/palletsprojects_flask/palletsprojects_flask-icon.svg" width="20"> Web framework for API development

- **Gunicorn** <img align="center" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/gunicorn.svg" width="20"> WSGI HTTP Server for production deployment

- **OpenCV** <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" width="20"> Computer vision and image processing

- **Scikit-learn** <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" width="20"> Machine learning algorithms

</div>

## Production Implementation
<div align="center">
    <img width="75%" src="https://github.com/user-attachments/assets/ddfaa06d-4d81-4162-8725-c74e7ea23a91" alt="Figma model" />
</div>

Making sure we have a complete production implementation was important to us during this hackathon. This included setting up a domain with a proper SSL certificate, and configuring it to a server that would keep up under higher intensity loads. By the end of the hackathon, we had successfully configured the following:

* A **t2.medium** AWS EC2 instance <img align="center" src="https://cdn.simpleicons.org/amazon" height="20">, to keep up with the machine learning model usage
* A valid HTTPS server with a domain (arrietty.org). This was accomplished using **Cloudflare registrar** <img src="https://cdn.simpleicons.org/cloudflare" height="20" align="center">
* Nginx <img align="center" src="https://cdn.simpleicons.org/nginx" height="20"> (the faster alternative to **Apache** <img align="center" src="https://cdn.simpleicons.org/apache" height="20"> or **Caddy** <img align="center" src="https://cdn.simpleicons.org/caddy" height="20">) as our reverse proxy, porting incoming HTTPS data from the domain to localhost:8000
* A Python <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="20"> WSGI HTTP Server (**Gunicorn** <img align="center" src="https://raw.githubusercontent.com/gilbarbara/logos/master/logos/gunicorn.svg" width="20">), allowing us to log info/errors, and maximize how many threads our Flask instance can run on
* A **Flask** <img align="center" src="https://www.vectorlogo.zone/logos/palletsprojects_flask/palletsprojects_flask-icon.svg" width="20"> instance running in production mode, routing different endpoints to their respective business logic
* Pre-trained data using **OpenCV** <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" width="20">, allowing our **Scikit Learn** <img align="center" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/scikitlearn/scikitlearn-original.svg" width="20"> ML model to convert input to output

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/raghav002/MadHacks2024Proj/
    cd MadHacks2024Proj
    ```
2. Ensure you have Python installed
   - While this step is dependent on your OS, you can check if it exists by typing: ```python3 --version``` or ```python --version```
   - If Python is not installed, you can view the installation at <a>https://www.python.org/downloads/</a>

3. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

    While this step is not required, it is recomended for lower dependency conflicts.
    
4. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Collect images for training the model (see [Data Collection](#data-collection)).
2. Train the model using the collected data (see [Training the Model](#training-the-model)).
3. Run the Flask application (see [Running the Application](#running-the-application)).

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

You can find the endpoints at [Endpoints](#endpoints)

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
