from gtts import gTTS
import pygame
import os

def speak(file_path="output.txt"):
    # Read the content of the file
    try:
        with open(file_path, 'r') as file:
            fixed_text = file.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return

    # Convert text to speech and save as mp3
    language = "en"
    myobj = gTTS(text=fixed_text, lang=language, slow=False)
    print(f"Speaking text: {fixed_text}")
    myobj.save("text.mp3")

    # Initialize pygame mixer
    pygame.mixer.init()
    try:
        # Load and play the audio file
        pygame.mixer.music.load("text.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until the music finishes playing
            continue
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
        # Quit the mixer and remove the file
        pygame.mixer.quit()
        if os.path.exists("text.mp3"):
            os.remove("text.mp3")