# tts.py

import subprocess
import threading
import time
import queue

tts_queue = queue.Queue()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        try:
            subprocess.call(["say", text])
        except Exception as e:
            print(f"[say Error] {e}")
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def speak(text):
    tts_queue.put(text)
