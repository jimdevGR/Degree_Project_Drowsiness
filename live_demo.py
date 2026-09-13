from __future__ import annotations # tells Python to store type hints as plain strings
import cv2
import tensorflow as tf
from math_funcs import calculate_ear, calculate_mar, left_eye_indices, right_eye_indices, mouth_indices # from user defined .py file
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
import numpy as np
import time
from termcolor import colored
from keras.saving import load_model
from collections import deque # double ended queue (a type of list)
from helper_funcs import MIN_EAR_RANGE, MIN_MAR_RANGE # 0.20, 0.25

# paths to models
mp_model_path = r"C:\MediapipeModel\face_landmarker.task"
my_model_path = r"C:\PROJECT_drowsiness\best_model_so_far\model(6fps_stride2).keras"

# global variables, the values are from train_model.py
windowSize = 30
windowStride = 2
smoothingSize = 13
best_threshold = 0.3753

latest_avg_ear = 0.0
latest_mar = 0.0
smoothed_probability = 0.0
stride_counter = 0
history_dequeSize = 180 # 30 sec
min_history_dequeSize = 60 # 10 sec
continuous_prob_awake = 0 
awake_prob_threshold = 0.3
continuous_prob_awake_threshold = 3

# create double ended queues, we can insert/delete elements from both sides
features_deque = deque(maxlen = windowSize)
probabilities_deque = deque(maxlen = smoothingSize)
history_deque = deque(maxlen = history_dequeSize) # we use 180 previous frames (30 sec) for effective min-max scaling based on the min-max ear/mar

# load keras model only for predictions, it's already compiled and trained 
my_model = load_model(filepath = my_model_path, compile = False)

# callback function, gets called automatically by mediapipe when she has the landmarks for the current frame ready
# uses type hints
def mp_async_func(mp_result: mp.tasks.vision.python.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_avg_ear, latest_mar, smoothed_probability, stride_counter, continuous_prob_awake
    if mp_result.face_landmarks: # if we have landmarks
        landmarks = mp_result.face_landmarks[0] # get the landmarks for the first face detected
        img_height, img_width = output_image.height, output_image.width
        ear_left = calculate_ear(landmarks, left_eye_indices, img_width, img_height, "Left")
        ear_right = calculate_ear(landmarks, right_eye_indices, img_width, img_height, "Right")
        avg_ear = (ear_left + ear_right) / 2
        mar = calculate_mar(landmarks, mouth_indices, img_width, img_height)
        latest_avg_ear = avg_ear
        latest_mar = mar
        # if deque is full, an element from the left of the deque will be discarded
        features_deque.append([avg_ear, mar]) # append to the right
        history_deque.append([avg_ear, mar])
        stride_counter += 1

        # implement a calibration phase
        if len(history_deque) < min_history_dequeSize: # if the history size is smaller than 60 frames (10 sec) we dont make a prediction
            return

        if stride_counter >= windowStride: # make a prediction every 2 frames (stride 2), we apply min-max scaling to the features
            stride_counter = 0
            history_2D = np.array(object = history_deque, dtype = np.float32) # keras uses numbers with type np.float32
            min_ear, min_mar = history_2D.min(axis = 0)
            max_ear, max_mar = history_2D.max(axis = 0)
            ear_denom = max(max_ear - min_ear, MIN_EAR_RANGE) # apply safety bounds
            mar_denom = max(max_mar - min_mar, MIN_MAR_RANGE)
            min_features = np.array(object = [min_ear, min_mar], dtype = np.float32) # shape (2, )
            denoms = np.array(object = [ear_denom, mar_denom], dtype = np.float32) # shape (2, )
            features_2D = np.array(object = features_deque, dtype = np.float32) # shape (30, 2)
            norm_features_2D = (features_2D - min_features) / denoms # normalize the features with vector operations
            input_array = norm_features_2D.reshape(1, windowSize, 2)
            input_tensor = tf.convert_to_tensor(value = input_array, dtype = np.float32) # turn ndarray to a tf.Tensor object
            output_tensor = my_model(input_tensor, training = False) # direct call (uses behind __call__ method) instance my_model to have predictions with less latency 
            prediction = float(output_tensor.numpy()[0][0])

# if the model gives at least 3 continuous raw low probabilities, so it's sure that the driver is awake, 
# we replace all the elements inside the probabilities queue with the last given probability, we do this to possibly minimize 
# the delay in the recognition of the alertness after drowsiness
            if prediction <= awake_prob_threshold: continuous_prob_awake += 1
            else: continuous_prob_awake = 0

            if continuous_prob_awake >= continuous_prob_awake_threshold: # after at least 1 sec of low raw probs
                probabilities_deque.clear()
                probabilities_deque.extend([prediction] * smoothingSize)
            else: probabilities_deque.append(prediction)

            if len(probabilities_deque) == smoothingSize: # if we collected smoothingSize probabilities
                smoothed_probability = float(np.median(probabilities_deque)) # find the median for every new raw probability in the window

# initialize the mediapipe task in live stream mode, runs in a background thread to 
# not delay the main processsing (camera loop and opencv) that runs in the 'main' thread
base_options = BaseOptions(model_asset_path = mp_model_path)
landmarker_options = FaceLandmarkerOptions(base_options = base_options,
                                           running_mode = RunningMode.LIVE_STREAM,
                                           num_faces = 1,
                                           output_face_blendshapes = False,
                                           output_facial_transformation_matrixes = False,
                                           result_callback = mp_async_func
                                           )

def main():
    with FaceLandmarker.create_from_options(landmarker_options) as my_landmarker:
        live_video_object = cv2.VideoCapture(0) # video capture object for the default camera
        frame_counter = 0
        start_time = time.time() # get the time in seconds since January 1, 1970, 00:00:00

        while live_video_object.isOpened(): # endless loop
            bool_value, numpy_frame = live_video_object.read()
            if not bool_value: break # exit loop if read() couldn't read a frame
            frame_counter += 1

            if frame_counter % 5 == 0: # process 1 in 5 frames(6fps)
                frame_timestamp_ms = int((time.time() - start_time) * 1000)
                rgb_numpy_frame = cv2.cvtColor(src = numpy_frame, code = cv2.COLOR_BGR2RGB) # opencv reads the images in the bgr color space
                mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_numpy_frame) # # turn the ndarray of the frame to an mp.Image object

                try:
                    # asynchronous call for detecting the face landmarks
                    my_landmarker.detect_async(mp_image, frame_timestamp_ms)
                except Exception as e:
                    print(colored(f"Mediapipe could not process frame with number ({frame_counter}): {e}", color = "yellow", attrs = ["bold", "italic"]))
                    print(colored("Continuing to the next frame...", color = "green", attrs = ["bold", "italic", "underline"]))

            if len(history_deque) < min_history_dequeSize:
                 calibration_progress = len(history_deque) / min_history_dequeSize
                 cv2.putText(img = numpy_frame, text = f"Calibration in progress... Progress: {calibration_progress:.2%}", org = (100, 450), \
                             fontFace = cv2.FONT_ITALIC, fontScale = 0.6, color = (0, 255, 255), thickness = 2, lineType = cv2.LINE_AA)
            else:
                 cv2.putText(img = numpy_frame, text = f"PROB: {smoothed_probability:.2%}", org = (450, 160), fontFace = cv2.FONT_ITALIC, \
                             fontScale = 0.8, color = (0, 0, 255), thickness = 2, lineType = cv2.LINE_AA)
            
            cv2.putText(img = numpy_frame, text = f"EAR: {latest_avg_ear:.3f}", org = (450, 70), fontFace = cv2.FONT_ITALIC, \
                        fontScale = 0.8, color = (255, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
            cv2.putText(img = numpy_frame, text = f"MAR: {latest_mar:.3f}", org = (450, 115), fontFace = cv2.FONT_ITALIC, \
                        fontScale = 0.8, color = (0, 255, 0), thickness = 2, lineType = cv2.LINE_AA)
            cv2.imshow(winname = f"Camera footage", mat = numpy_frame)

            # wait for 1 ms, if the user pressed the letter 'q' exit the camera loop
            if (cv2.waitKey(1) & 0xFF) == ord('q'): 
                break

        live_video_object.release()

# main() executes only when the whole script is ran directly but not when it's imported
if __name__ == "__main__": 
    main()
