import cv2
import os
from termcolor import colored
from shutil import rmtree
from math_funcs import calculate_ear, calculate_mar, left_eye_indices, right_eye_indices, mouth_indices
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions

frames = [r"C:\PROJECT_drowsiness\frames\frame_0.jpg", r"C:\PROJECT_drowsiness\frames\frame_11.jpg", r"C:\PROJECT_drowsiness\frames\frame_40.jpg"]
model_path = r"C:\MediapipeModel\face_landmarker.task"
folder_to_save = r"C:\PROJECT_drowsiness\frames_with_landmarks"

# function to draw landmarks on frame
def draw_landmarks(input_frame, mp_landmarks, indices, frame_height, frame_width, landmarks_color):
    for index in indices:
        landmark = mp_landmarks[index]
        landmark_x, landmark_y = int(landmark.x * frame_width), int(landmark.y * frame_height) # cv2.circle wants the points as type int
        cv2.circle(input_frame, center = (landmark_x, landmark_y), radius = 4, color = landmarks_color, thickness = 1)

# initialize and create the mediapipe task
base_options = BaseOptions(model_asset_path = model_path)
landmarker_options = FaceLandmarkerOptions(base_options = base_options,
                                           num_faces = 1,
                                           output_face_blendshapes = False,
                                           output_facial_transformation_matrixes = False,
                                           )

try:
    rmtree(folder_to_save)
    print(colored(f"Folder '{os.path.basename(folder_to_save)}' deleted successfully!", color = "green", attrs = ["bold", "italic"]))
except FileNotFoundError:
    print(colored(f"Folder '{os.path.basename(folder_to_save)}' not found!", color = "red", attrs = ["bold", "italic"]))
except PermissionError:
    print(colored(f"You don't have rights to delete folder '{os.path.basename(folder_to_save)}'!", color = "red", attrs = ["bold", "italic"]))
except Exception as e:
    print(colored(f"Error occurred: {e}", color = "yellow", attrs = ["bold", "italic"]))

try:
    if not os.path.exists(folder_to_save):
        os.makedirs(folder_to_save)
except OSError:
    print (colored(f"Error creating directory '{os.path.basename(folder_to_save)}': {e}", color = "red", attrs = ["bold", "italic"]))

def main():
    with FaceLandmarker.create_from_options(landmarker_options) as my_landmarker:
        for frame_path in frames:
            frame = cv2.imread(frame_path)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2RGB) # color space conversion
            mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame) # turn the frame to an mp.Image object
            result = my_landmarker.detect(mp_frame)
            
            if result.face_landmarks:
                landmarks = result.face_landmarks[0] # get the landmarks of the first recognised face
                ear_left = calculate_ear(landmarks, left_eye_indices, width, height, "Left")
                ear_right = calculate_ear(landmarks, right_eye_indices, width, height, "Right")
                avg_ear = (ear_left + ear_right) / 2
                mar = calculate_mar(landmarks, mouth_indices, width, height)
                draw_landmarks(frame, landmarks, left_eye_indices + right_eye_indices, height, width, (255, 0, 0)) # draw eye points with blue
                draw_landmarks(frame, landmarks, mouth_indices, height, width, (0, 0, 255)) # draw mouth points with red
                cv2.putText(img = frame, text = f"EAR: {avg_ear:.3f}", org = (550, 80), fontFace = cv2.FONT_ITALIC, \
                            fontScale = 1, color = (255, 0, 0), thickness = 2, lineType = cv2.LINE_AA)
                cv2.putText(img = frame, text = f"MAR: {mar:.3f}", org = (550, 125), fontFace = cv2.FONT_ITALIC, \
                            fontScale = 1, color = (0, 0, 255), thickness = 2, lineType = cv2.LINE_AA)
                cv2.imshow(f"{os.path.basename(frame_path)}", frame) # show image with landmarks
                name = fr"{folder_to_save}\lm_{os.path.basename(frame_path)}"
                cv2.imwrite(name, frame)
                print(colored(f"{os.path.basename(frame_path)} --> EAR: {avg_ear:.3f} | MAR: {mar:.3f}", color = "green", attrs = ["bold", "italic"]))
            
                cv2.waitKey(0) # wait until any key is pressed to close the windows

if __name__ == "__main__":
    main()







        


