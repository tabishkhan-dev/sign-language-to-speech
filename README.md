# Sign Language to Speech

This project converts sign language gestures into speech using deep learning and computer vision.  
It uses MediaPipe for hand tracking, TensorFlow/Keras for classification, and a Text-to-Speech (TTS) module for audio output.  

---

## Features
- Detects and recognizes American Sign Language (ASL) hand gestures (A–Z).
- Converts recognized gestures into corresponding text.
- Outputs the text as speech in real-time.
- Modular code: dataset collection, training, recognition, and TTS are separate scripts.

---

## Installation

STEP 1: CLONE THE REPOSITORY  
git clone https://github.com/tabishkhan-dev/sign-language-to-speech.git  
cd sign-language-to-speech  

STEP 2: CREATE AND ACTIVATE A VIRTUAL ENVIRONMENT  
python -m venv venv  
source venv/bin/activate   # Mac/Linux  
venv\Scripts\activate      # Windows  

STEP 3: INSTALL DEPENDENCIES  
pip install -r requirements.txt  

STEP 4: RUN THE APPLICATION  
python main.py  

---

## Usage

CAPTURE NEW DATASET  
python capture_dataset.py  

TRAIN CUSTOM MODEL  
python train_custom_model.py  

TRAIN ASL MODEL  
python train_asl_model.py  

RUN REAL-TIME GESTURE RECOGNITION  
python gesture_recognition.py  

RUN TEXT TO SPEECH  
python tts.py  

---

## Project Structure
├── dataset/                # Collected gesture images  
├── asl_classifier.py       # Model for classification  
├── asl_model.h5            # Pretrained ASL model  
├── asl_custom_model.h5     # Custom trained model  
├── capture_dataset.py      # Script to collect dataset  
├── train_custom_model.py   # Training script (custom model)  
├── train_asl_model.py      # Training script (ASL model)  
├── gesture_recognition.py  # Real-time gesture recognition  
├── tts.py                  # Text-to-speech module  
├── label_map.txt           # Class labels  
├── main.py                 # Main execution file  
└── README.md               # Documentation  

---

## Dataset## Dataset
⚠️ The training dataset (SignMNIST) is not included in this repository due to size limits.  

- Pre-trained models (`asl_model.h5`, `asl_custom_model.h5`) are already included, so the project runs without the dataset.  
- If you want to re-train the models, you can download the dataset from [Kaggle: Sign Language MNIST](https://www.kaggle.com/datamunge/sign-language-mnist).  


---

## Contributing
Feel free to fork this repository, submit pull requests, or open issues for suggestions and improvements.

---

## License
This project is licensed under the MIT License.
