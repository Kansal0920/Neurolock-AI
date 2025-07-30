from gtts import gTTS
import os
import tempfile
import time

def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            os.system(f"afplay {fp.name}")  # MacBook uses afplay
            time.sleep(0.5)  # brief delay to ensure playback
            os.remove(fp.name)
    except Exception as e:
        print("❌ Voice Feedback Error:", e)
