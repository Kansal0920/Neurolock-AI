# generate_alert.py
from gtts import gTTS

text = "Suspicious Emotion Detected!"
tts = gTTS(text=text, lang='en')
tts.save("voice_alert.mp3")

print("✅ voice_alert.mp3 generated successfully!")
