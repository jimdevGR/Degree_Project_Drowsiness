import cv2
import os
import csv
from glob import glob
from termcolor import colored
from math_funcs import calculate_ear, calculate_mar, left_eye_indices, right_eye_indices, mouth_indices # from user defined .py file
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

model_path = r"C:\MediapipeModel\face_landmarker.task"
active_folder_path = r"C:\PROJECT_drowsiness\raw_videos\active"
sleepy_folder_path = r"C:\PROJECT_drowsiness\raw_videos\sleepy"
path_to_csv = r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv"
video_extensions = ["*.avi", "*.mp4"]
active_video_files = []
sleepy_video_files = []
all_videos = []
video_id = 1 # it will be used for distinguishing the video files in the final .csv file
csv_col_headers = ["Video Identifier", "Video Name", "Frame Number", "Timestamp", "EAR", "MAR", "Subject", "State"]

# get all the videos with active and sleepy individuals 
for extension in video_extensions:
    active_video_files.extend(glob(os.path.join(active_folder_path, extension)))
    sleepy_video_files.extend(glob(os.path.join(sleepy_folder_path, extension)))

# all_videos list contains elements of form (video path, "State")
all_videos = [(v_path, "Active") for v_path in active_video_files] + [(v_path, "Sleepy") for v_path in sleepy_video_files]

# initialize and create the mediapipe task
# note that mediapipe automatically rotates, resizes, normalizes and converts the color space of the frames
base_options = BaseOptions(model_asset_path = model_path)
landmarker_options = FaceLandmarkerOptions(base_options = base_options,
                                           running_mode = RunningMode.VIDEO,
                                           num_faces = 1,
                                           output_face_blendshapes = False,
                                           output_facial_transformation_matrixes = False,
                                           )

def main():
    global video_id
    with open(path_to_csv, mode = "w", newline = '') as csv_file: 
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_col_headers) # write headers to csv
        
        print(colored("Processing of the videos in progress, please wait...", color = "blue", attrs = ["bold", "italic"]))

        # loop for processing all the videos and preparing the data for the lstm
        for video_path, state in all_videos:
            # create a FaceLandmarker obj for every video
            with FaceLandmarker.create_from_options(landmarker_options) as my_landmarker:
                video_object = cv2.VideoCapture(video_path)
                frame_counter = 0
                fps = video_object.get(cv2.CAP_PROP_FPS) # get the fps of the video
                if fps <= 0: fps = 30.0 # fallback value
                
                if not video_object.isOpened():
                    print(colored(f"Failed to open video file with name {os.path.basename(video_path)} and state {state}!", color = "red", attrs = ["bold", "italic"]))
                    print(colored("Continuing to the next video...", color = "green", attrs = ["bold", "italic"]))
                    continue # continue to the next video
                
                while video_object.isOpened(): # loop for all the frames of the video
                    bool_value, frame = video_object.read()
                    
                    if not bool_value:
                        break # exit current video
                    
                    if frame_counter % 5 == 0: # process every 1 in 5 frames(6fps), skip 4, for less cpu usage
                        frame_timestamp_ms = int((frame_counter / fps) * 1000) # mediapipe needs the timestamp in ms and in type int
                        height, width = frame.shape[:2]
                        rgb_frame = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2RGB) # opencv reads the images in the bgr color space
                        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame) # turn the ndarray of the frame to an mp.Image object
                       
                        try: # try-except to protect mediapipe from possible ruined frames
                            result = my_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
                            if result.face_landmarks: # if we have landmarks
                                # face_landmarks is a list of lists for every face in the frame, we want the first one
                                landmarks = result.face_landmarks[0]
                                ear_left = calculate_ear(landmarks, left_eye_indices, width, height, "Left")
                                ear_right = calculate_ear(landmarks, right_eye_indices, width, height, "Right")
                                avg_ear = (ear_left + ear_right) / 2
                                mar = calculate_mar(landmarks, mouth_indices, width, height)
                                file_name = os.path.basename(video_path)
                                subject_id = file_name.split("-")[0]
                                csv_row = [f"Video_{video_id:03d}", f"{file_name}", frame_counter, frame_timestamp_ms / 1000, avg_ear, mar, subject_id]
                                csv_row.append(0 if state == "Active" else 1)
                                csv_writer.writerow(csv_row)
                        except Exception as e:
                            print(colored(f"Mediapipe could not process frame {frame_counter} of video with path {video_path}: {e}", color = "yellow", attrs = ["bold", "italic"]))
                    frame_counter += 1
                video_id += 1
                video_object.release() # release resources obtained for the current video
            print("****************************************************************************************")
            print(colored(f"Processing of video with path {video_path} ended successfully!", color = "green", attrs = ["bold", "italic"])) 
            print("****************************************************************************************")
             
if __name__ == "__main__":
    main()


