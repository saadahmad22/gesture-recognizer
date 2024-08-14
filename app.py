import os
import tempfile
import cv2
import json
import base64
from flask import Flask, request, jsonify
import mediapipe as mp
from gesture_recognition_model import GestureRecognitionModel, default_options
from config import port, secret_key, host

app = Flask(__name__)
app.secret_key = secret_key

@app.route('/predict', methods=['POST'])
def predict():
    ''' Predicts the gesture from the input data

    Website to encode JSON Strings to BASE64: https://www.base64encode.org/
        - Use to configure model_configs
    
    Args:
        Multipart form-data with the following format:
        {
            "gesture": file
            model_configs: JSON Object as BASE64 string with the following optional fields: {
                "running_mode": str,
                "num_hands": int,
                "min_hand_detection_confidence": float,
                "min_hand_presence_confidence": float,
                "min_tracking_confidence": float,
                "canned_gestures_classifier_options": {
                    "display_names_locale": str,
                    "max_results": int,
                    "score_threshold": int,
                    "category_allowlist": list[str],
                    "category_denylist": list[str]      
                },
                "custom_gestures_classifier_options": {
                    "display_names_locale": str,
                    'max_results': int,
                    'score_threshold': int,
                    'category_allowlist': list[str],
                    'category_denylist': list[str]      
                },
                "result_callback": callable
                "image_options": {
                    "region_of_interest": list[float], len=4
                    "rotation_degrees": int, divisible by 90
                }
            }
        }
    '''        
    
    try:
        configs = json.loads(base64.b64decode(request.form.get("model_configs")).decode('utf-8')) if request.form.get("model_configs") else {}

        options = {
            "running_mode": "IMAGE",
            "num_hands": int(configs.get("num_hands", default_options["num_hands"])),
            "min_hand_detection_confidence": float(configs.get("min_hand_detection_confidence", default_options["min_hand_detection_confidence"])),
            "min_hand_presence_confidence": float(configs.get("min_hand_presence_confidence", default_options["min_hand_presence_confidence"])),
            "min_tracking_confidence": float(configs.get("min_tracking_confidence", default_options["min_tracking_confidence"])),
            "canned_gestures_classifier_options": configs.get("canned_gestures_classifier_options", default_options["canned_gestures_classifier_options"]),
            "custom_gestures_classifier_options": configs.get("custom_gestures_classifier_options", default_options["custom_gestures_classifier_options"]),
            "result_callback": configs.get("result_callback", default_options["result_callback"])            
        }
        model = GestureRecognitionModel(**options)
        mode = configs.get("running_mode", default_options["running_mode"])
        print(model.options)

        if 'gesture' not in request.files:
            return jsonify({"error": "No file found"}), 400

        file = request.files['gesture']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = os.path.join(tmpdirname, file.filename)
            file.save(file_path)
            image_options = configs.get('image_options', None)
            if mode == 'IMAGE':
                image = mp.Image.create_from_file(file_path)
                # Process the image for gesture recognition
                prediction = model.predict(image, image_processing_options=image_options)
            elif mode in ['VIDEO', 'STREAM']:
                # Read the file using cv2
                video = cv2.VideoCapture(file_path)
                frame_rate = video.get(cv2.CAP_PROP_FPS)
                prediction = []
                i = 0
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break
                    file_path = f"{tmpdirname}/frame_{i}.png"
                    cv2.imwrite(file_path, frame)

                    prediction.append(
                        model.predict(
                            mp.Image.create_from_file(file_path),
                            frame_timestamp_ms=1000 * i // frame_rate,
                            image_processing_options=image_options
                            )                            
                        )
                    i += 1
                video.release()
            else:
                return jsonify({"error": "Invalid mode"}), 400

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host=host, port=port, debug=True)