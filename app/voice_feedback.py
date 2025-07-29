# app/voice_feedback.py

import pygame
import time
import pyttsx3

# Initialize TTS engine (offline, no API needed)
engine = pyttsx3.init()
engine.setProperty('rate', 160)  # Adjust voice speed

def speak_alert(text):
    print(f"[NEUROLOCK] ALERT: {text}")
    engine.say(text)
    engine.runAndWait()
