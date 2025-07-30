import os
from datetime import datetime

def generate_alert(emotion):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{now}] 🔒 Access Denied: Suspicious emotion detected - '{emotion}'"
    print(message)

    # Optional: play alert sound (ensure sound.mp3 exists or remove this block)
    try:
        from playsound import playsound
        playsound("assets/alert.mp3")  # Make sure the path & file exist
    except Exception as e:
        print("🔇 Sound alert skipped:", e)

    # Log threat
    with open("threat_log.txt", "a") as log:
        log.write(message + "\n")

    # Optionally, trigger further action (lock system, notify, etc.)
    # os.system("shutdown /l")  # Logout command (Windows only)
