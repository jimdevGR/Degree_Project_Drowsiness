import matplotlib.pyplot as plt
import cv2
from termcolor import colored
import os

dataset_videos = [
                 r"C:\PROJECT_drowsiness\raw_videos\active\1-FemaleNoGlasses-Normal.avi",
                 r"C:\PROJECT_drowsiness\raw_videos\active\2-FemaleNoGlasses-Normal.avi",
                 r"C:\PROJECT_drowsiness\raw_videos\active\20-MaleGlasses-Normal.avi",
                 r"C:\PROJECT_drowsiness\raw_videos\sleepy\uta_microsleep_7.mp4",
                 r"C:\PROJECT_drowsiness\raw_videos\sleepy\front_microsleep.mp4",
                 r"C:\PROJECT_drowsiness\raw_videos\active\40-MaleNoGlasses-Normal.avi"
                 ]

video_frames = []
counter = 1

for video_path in dataset_videos:
    video_object = cv2.VideoCapture(video_path)

    if not video_object.isOpened():
        print(colored(f"Failed to open video file with name {os.path.basename(video_path)}!", color = "red", attrs = ["bold", "italic"]))
        print(colored("Continuing to the next video...", color = "green", attrs = ["bold", "italic"]))
        continue # continue to the next video

    bool_value, frame = video_object.read()
    rgb_frame = cv2.cvtColor(src = frame, code = cv2.COLOR_BGR2RGB)
    video_frames.append(rgb_frame)

# plot some images examples
plt.figure(num = "Dataset Images Examples", figsize = (7, 5), facecolor = "darkgrey", edgecolor = "black", linewidth = 2, layout = "tight")
for frame in video_frames:
    plt.subplot(2, 3, counter)
    plt.xticks(ticks = [])
    plt.yticks(ticks = [])
    plt.grid(visible = False)
    plt.box(on = False)
    plt.imshow(X = frame)
    plt.xlabel(f"({counter})")
    counter += 1
plt.show()






