# Degree_Project_Drowsiness
---THIS IS A DEGREE PROJECT A DROWSINESS DETECTION SYSTEM---
The repository contains the following files:
data_preprocessing.py: This python script takes the raw videos and uses opencv and the mediapipe model to tranform them into the frames_data.csv. Uses the math_funcs.py script too.
math_funcs: This python script contains two functions, one for calculating the EAR from the landmarks mediapipe gave for the frame and one for calculating the MAR.
extract_frames.py: This python script takes a selected video from the dataset and extracts all the frames in it using opencv. It saves all those frames to the computer.
draw_landmarks: This python script takes some frames extracted from the previous file and using opencv, mediapipe and a custom function it draws circles to the positions of the landmarks that mediapipe gave for these frames and are used for calculating the EAR and MAR. Was created for visualization purposes.
plot_images.py: This python script just takes some selected videos from the dataset and plots their first frame to a simple matplotlib plot. Was also created for visualization purposes.
helper_funcs.py: This python script contains three functions that are used from the train_model.py. One splits the dataset to train, test and validation sets, one tranforms the data of the three sets to "windows" and one who applies smoothing onto the predictions the model gives to correct mistakes.
train_model.py This python script prepares the data for the model using functions inside the helper_funcs.py module, defines the model's architecture, trains it, evaluates it's perfomance, takes some predictions from it and makes a heatmap for visualizing the predictions.
