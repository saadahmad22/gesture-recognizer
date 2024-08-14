# Gesture Recognizer
Project integrating a pre-trained gesture recognition model into a sample Flask backend API structure

## Introduction

## Setup

Note, this project was developed using Python 3.12

This project is web interface for Google's MediaPipe Gesture Recognizer
Link: https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer

### 1. It is recommended install a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

use ```deactivate``` to turn the environment off
### 2. Install project dependencies

Developed on python pip version 24.0

```bash
pip install -r requirements.txt
```

### 3. Run the flask application
Edit the `config.py` file as needed

```bash
flask run
```

### 4. Postman folder contains importable JSON
Import and run collection
NOTE: must upload image and video files to respective API test calls


#### Parameter structure:
    - "gesture": Image/video file
    - "model_configs": Takes Base64 encoded JSON object as outlined in the function documentation
        - e.g., Sample for image
            - Source `{"running_mode":"IMAGE","num_hands":2,"min_hand_detection_confidence":0.5,"min_hand_presence_confidence":0.5,"min_tracking_confidence":0.5}`
            - Use https://www.base64encode.org to generate Base64 encoded string, as seen below for above source JSON `eyJydW5uaW5nX21vZGUiOiJJTUFHRSIsIm51bV9oYW5kcyI6MiwibWluX2hhbmRfZGV0ZWN0aW9uX2NvbmZpZGVuY2UiOjAuNSwibWluX2hhbmRfcHJlc2VuY2VfY29uZmlkZW5jZSI6MC41LCJtaW5fdHJhY2tpbmdfY29uZmlkZW5jZSI6MC41fQ==`
        - Video sample
            - Source `eyJydW5uaW5nX21vZGUiOiJWSURFTyIsIm51bV9oYW5kcyI6MiwibWluX2hhbmRfZGV0ZWN0aW9uX2NvbmZpZGVuY2UiOjAuNSwibWluX2hhbmRfcHJlc2VuY2VfY29uZmlkZW5jZSI6MC41LCJtaW5fdHJhY2tpbmdfY29uZmlkZW5jZSI6MC41fQ==`
            - Use https://www.base64encode.org to generate Base64 encoded string, as seen below for above source JSON 
            `eyJydW5uaW5nX21vZGUiOiJWSURFTyIsIm51bV9oYW5kcyI6MiwibWluX2hhbmRfZGV0ZWN0aW9uX2NvbmZpZGVuY2UiOjAuNSwibWluX2hhbmRfcHJlc2VuY2VfY29uZmlkZW5jZSI6MC41LCJtaW5fdHJhY2tpbmdfY29uZmlkZW5jZSI6MC41fQ==`
        - parameters can be customized as per Google mediapipe Gesture Recognizer API documentation `https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/GestureRecognizerOptions`


## Contact for issues/questions
Name: Saad Ahmad

Email: saadahmad9@outlook.com
