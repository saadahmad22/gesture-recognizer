from typing import Optional, Callable, Iterable

import mediapipe as mp
import mediapipe.tasks as mp_tasks
from mediapipe.tasks.python.components.processors import classifier_options
from config import model_path
import cv2
print(cv2.__version__)

# load classes from mediapipe
BaseOptions = mp_tasks.BaseOptions
GestureRecognizer = mp_tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp_tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp_tasks.vision.RunningMode


default_options = {
    "running_mode": "IMAGE",
    "num_hands": 1,
    "min_hand_detection_confidence": 0.5,
    "min_hand_presence_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "canned_gestures_classifier_options": None,
    "custom_gestures_classifier_options": None,
    "result_callback": None
}

# utility functions
def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)

def validate_classifier_options(options: dict | classifier_options.ClassifierOptions) -> classifier_options.ClassifierOptions:
    ''' Validates the classifier options and returns the default options if not provided.'''

    if not options:
        options = {}
    elif isinstance(options, classifier_options.ClassifierOptions):
        return options
    
    return_obj = classifier_options.ClassifierOptions()
    for key in ['display_names_locale', 'max_results', 'score_threshold', 'category_allowlist', 'category_denylist']:
        if key in options:
            return_obj.__dict__[key] = options[key]
    return return_obj

def validate_image_processing_options(image_processing_options: Optional[dict[str, Iterable[int] | int]]) -> mp_tasks.vision.ImageProcessingOptions | None:
    ''' Validates the image processing options and returns the default options if not provided.'''

    # only need to validate if it exists, else it is fine
    if image_processing_options:
        if 'region_of_interest' in image_processing_options:
            region_of_interest = image_processing_options['region_of_interest']
            region_of_interest = mp_tasks.components.containers.Rect(*[clip(value, 0.0, 1.0) for value in region_of_interest])
            image_processing_options['region_of_interest'] = region_of_interest
        if 'rotation_degrees' not in image_processing_options:
            image_processing_options['rotation_degrees'] = 0
        return_obj = mp_tasks.vision.ImageProcessingOptions()
        return_obj.region_of_interest = image_processing_options.get('region_of_interest', None)
        return_obj.rotation_degrees = image_processing_options['rotation_degrees']
        image_processing_options = return_obj
    return image_processing_options

class GestureRecognitionModel:
    '''GestureRecognitionModel class to perform gesture recognition on images, videos, or live streams.'''

    def __init__(self, *args, **kwargs):
        ''' Initializes the GestureRecognitionModel class with the given options.

        See create_from_options() for more details on the options.
        '''

        self.create_from_options(*args, **kwargs)

    def create_from_options(
            self, 
            running_mode: str="IMAGE", 
            num_hands: int=1, 
            min_hand_detection_confidence: float=.5, 
            min_hand_presence_confidence: float=.5,
            min_tracking_confidence: float=.5,
            canned_gestures_classifier_options: dict | classifier_options.ClassifierOptions | None =None,
            custom_gestures_classifier_options: dict | classifier_options.ClassifierOptions | None=None,
            result_callback: Optional[Callable[[mp_tasks.vision.GestureRecognizerResult, mp.Image, int], None]]=None) -> None:
        ''' Creates a gesture recognizer instance with the given options.
        
        Args:
            running_mode: The running mode of the gesture recognizer.
                - "IMAGE": The recognizer runs on a single image.
                - "VIDEO": The recognizer runs on a sequence of images.
                - "STREAM": The recognizer runs on a stream of images.
            num_hands: The maximum number of hands can be detected by the GestureRecognizer.
                - Value in the range (0, inf)
            min_hand_detection_confidence: The minimum confidence score for the hand detection to be considered successful in palm detection model.
                - Values in the range [0.0, 1.0]
            min_hand_presence_confidence: The minimum confidence score of hand presence score in the hand landmark detection model. In Video mode and Live stream mode of Gesture Recognizer, if the hand presence confident score from the hand landmark model is below this threshold, it triggers the palm detection model. Otherwise, a lightweight hand tracking algorithm is used to determine the location of the hand(s) for subsequent landmark detection.
                - Values in the range [0.0, 1.0]
            min_tracking_confidence: The minimum confidence score for the hand tracking to be considered successful. This is the bounding box IoU threshold between hands in the current frame and the last frame. In Video mode and Stream mode of Gesture Recognizer, if the tracking fails, Gesture Recognizer triggers hand detection. Otherwise, the hand detection is skipped.
                - Values in the range [0.0, 1.0]
            canned_gestures_classifier_options: Options for configuring the canned gestures classifier behavior. The canned gestures are ["None", "Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Down", "Thumb_Up", "Victory", "ILoveYou"]
                - Display names locale: the locale to use for display names specified through the TFLite Model Metadata, if any.
                - Max results: the maximum number of top-scored classification results to return. If < 0, all available results will be returned.
                - Score threshold: the score below which results are rejected. If set to 0, all available results will be returned.
                - Category allowlist: the allowlist of category names. If non-empty, classification results whose category is not in this set will be filtered out. Mutually exclusive with denylist.
                - Category denylist: the denylist of category names. If non-empty, classification results whose category is in this set will be filtered out. Mutually exclusive with allowlist.
            custom_gestures_classifier_options: Options for configuring the custom gestures classifier behavior.
                - Display names locale: the locale to use for display names specified through the TFLite Model Metadata, if any.
                - Max results: the maximum number of top-scored classification results to return. If < 0, all available results will be returned.
                - Score threshold: the score below which results are rejected. If set to 0, all available results will be returned.
                - Category allowlist: the allowlist of category names. If non-empty, classification results whose category is not in this set will be filtered out. Mutually exclusive with denylist.
                - Category denylist: the denylist of category names. If non-empty, classification results whose category is in this set will be filtered out. Mutually exclusive with allowlist.
            result_callback: Sets the result listener to receive the classification results asynchronously when the gesture recognizer is in the live stream mode. Can only be used when running mode is set to LIVE_STREAM
                - Value is none or of type Callable[[mp_tasks.vision.GestureRecognizerResult, mp.Image, int]
                - MUST be provided if using a STREAM running mode.
        '''

        # value validation
        if not running_mode:
            running_mode = "IMAGE"
        match running_mode.upper().strip():
            case "IMAGE":
                self.running_mode = VisionRunningMode.IMAGE
            case "VIDEO":
                self.running_mode = VisionRunningMode.VIDEO
            case "STREAM":
                self.running_mode = VisionRunningMode.LIVE_STREAM
            case _:
                self.running_mode = VisionRunningMode.IMAGE
        
        # Create a gesture recognizer instance with the image mode:
        self.options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=self.running_mode,
            num_hands=clip(num_hands, 1, float("inf")),
            min_hand_detection_confidence=clip(min_hand_detection_confidence, 0.0, 1.0),
            min_tracking_confidence=clip(min_tracking_confidence, 0.0, 1.0),
            min_hand_presence_confidence=clip(min_hand_presence_confidence, 0.0, 1.0),
            canned_gesture_classifier_options=validate_classifier_options(canned_gestures_classifier_options),
            custom_gesture_classifier_options=validate_classifier_options(custom_gestures_classifier_options),
            result_callback=result_callback)        
        self.update_config()

    def update_config(self) -> None:
        ''' Updates the gesture recognizer model.'''

        self.close()
        self.gesture_recognizer = GestureRecognizer.create_from_options(self.options)

    def predict(
            self, 
            input_data: mp.Image, 
            frame_timestamp_ms: int=0,
            image_processing_options: Optional[dict[str, Iterable[int] | int]] = None) -> mp_tasks.vision.GestureRecognizerResult | None:
        ''' Predicts the gesture in the given image.

        Args:
            input_data: The input image for gesture recognition.                
            frame_timestamp_ms: The timestamp of the frame in milliseconds.
                - Value in the range [0, inf)
                - ONLY gets used when running mode is set to VIDEO.
                    - However, it is a MUST for this mode.
            image_processing_options: The image processing options.
                - Value is a dictionary with the following
                    - region_of_interest: The region of interest to process.
                        - default value is unspecified, meaning the entire image
                        - Value is a list of 4 floats representing the region of interest in the format [left, top, right, bottom]
                            - Coordinates must be in [0,1] with 'left' < 'right' and 'top' < 'bottom'.
                    - rotation_degrees: The rotation to apply to the image (or cropped region-of-interest), in degrees clockwise. 
                        - The rotation must be a multiple (positive or negative) of 90°.
                

        Returns:
            - If running in Image mode, returns the gesture recognition result. 
            - If running in Stream mode, returns None.
        '''        

        match self.running_mode:
            case VisionRunningMode.VIDEO:
                return self.gesture_recognizer.recognize_for_video(input_data, frame_timestamp_ms, validate_image_processing_options(image_processing_options))
            case VisionRunningMode.LIVE_STREAM:
                self.gesture_recognizer.recognize(input_data, validate_image_processing_options(image_processing_options))
                return None
            case _:
                return self.gesture_recognizer.recognize(input_data, validate_image_processing_options(image_processing_options))
    
    def get_running_mode(self) -> str:
        ''' Returns the running mode of the gesture recognizer.

        Returns:
            The running mode of the gesture recognizer.
        '''

        match self.running_mode:
            case VisionRunningMode.IMAGE:
                return "IMAGE"
            case VisionRunningMode.VIDEO:
                return "VIDEO"
            case VisionRunningMode.LIVE_STREAM:
                return "STREAM"
            case _:
                return "Unknown"        
            
    def close(self) -> None:
        ''' Closes the gesture recognizer instance.'''

        if hasattr(self, "gesture_recognizer"):
            self.gesture_recognizer.close()

    def set_mode(self, running_mode: str="IMAGE") -> None:
        ''' Sets the mode to process the gestures at the API level.

        Args:
            mode: The mode to process the gestures.
                - "IMAGE": The recognizer runs on a single image.
                - "VIDEO": The recognizer runs on a sequence of images.
                - "STREAM": The recognizer runs on a stream of images.
        '''

        if not running_mode:
            running_mode = "IMAGE"
        match running_mode.upper().strip():
            case "IMAGE":
                self.running_mode = VisionRunningMode.IMAGE
            case "VIDEO":
                self.running_mode = VisionRunningMode.VIDEO
            case "STREAM":
                self.running_mode = VisionRunningMode.LIVE_STREAM
            case _:
                self.running_mode = VisionRunningMode.IMAGE
        self.options.running_mode = self.running_mode