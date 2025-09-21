# Real-Time Sign Language Recognition (ASL + Dynamic HELLO Gesture)

## 📌 Overview
This project implements a **real-time sign language recognition system** that detects both **static ASL letters** and a **dynamic HELLO gesture**.  
It combines **MediaPipe** for hand tracking, a **custom-trained CNN** for static gesture classification, and a **motion-based algorithm** for the HELLO gesture.  
The system is lightweight, runs on standard hardware, and features a **projector-friendly user interface** with a sentence builder and optional **text-to-speech** output.

---

## 🎯 Features
- Recognition of **5 ASL letters**: `A`, `B`, `L`, `V`, `Y`
- Detection of **dynamic HELLO gesture** using palm-open + left-to-right motion tracking
- Real-time performance (~25–30 FPS on a MacBook Pro webcam)
- High-contrast UI with:
  - ROI box for consistent gesture placement
  - Confidence bar and score display
  - Sentence builder
- Keyboard shortcuts:
  - `SPACE` → Add space  
  - `D` → Delete last letter  
  - `S` → Speak sentence  
  - `ESC` → Exit

---


---

## ⚙️ Installation

1. **Clone the repository**:
```bash
git clone https://gitlab.tu-ilmenau.de/<username>/<projectname>.git
cd <projectname>

python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt

running the application
python main.py


