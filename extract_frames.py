import cv2
import os
from shutil import rmtree
from termcolor import colored

video_path = r"C:\PROJECT_drowsiness\raw_videos\sleepy\24-FemaleNoGlasses-Yawning_1.mp4"
images_path = r"C:\PROJECT_drowsiness\frames"
counter = 0

video_object = cv2.VideoCapture(video_path)

try:
    rmtree(images_path)
    print(colored(f"Folder '{os.path.basename(images_path)}' deleted successfully!", color = "green", attrs = ["bold", "italic"]))
except FileNotFoundError:
    print(colored(f"Folder '{os.path.basename(images_path)}' not found!", color = "red", attrs = ["bold", "italic"]))
except PermissionError:
    print(colored(f"You don't have rights to delete folder '{os.path.basename(images_path)}'!", color = "red", attrs = ["bold", "italic"]))
except Exception as e:
    print(colored(f"Error occurred: {e}", color = "yellow", attrs = ["bold", "italic"]))

try:
    if not os.path.exists(images_path):
        os.makedirs(images_path)
except OSError:
    print (colored(f"Error creating directory '{os.path.basename(images_path)}': {e}", color = "red", attrs = ["bold", "italic"]))

while video_object.isOpened():

    bool_value,frame = video_object.read()

    if bool_value:
        # if video is still left continue creating images
        name = fr"{images_path}\frame_" + str(counter) + ".jpg"
        print('Creating...' + name)
    
        # writing the extracted images
        cv2.imwrite(name, frame)
        counter += 1
    else:
        break

video_object.release()