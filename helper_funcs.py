import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from termcolor import colored

# global static variables for min-max scaling
MIN_EAR_RANGE = 0.20 # based on the 10th percentile of the ear ranges of the dataset videos
MIN_MAR_RANGE = 0.25 # based on the median range of the mar ranges

# function to split dataset to train and test sets based on subjects
def split_df(df, test_size_percentage, val_or_test):
    df_spliter = GroupShuffleSplit(n_splits = 1000, test_size = test_size_percentage, random_state = 1) # create a GroupShuffleSplit object
    current_df_drowsiness_ratio = df["State"].mean()
    split_found = False # flag to check whether a split was found
    
    # generate random dataset splits based on the 'Subject' column until we find the first one that satisfies our criteria
    # basically we use a random search approach
    for train_idxs, test_idxs in df_spliter.split(df, groups = df["Subject"]):
        test_df = df.iloc[test_idxs].reset_index(drop = True) # get the rows for the test dataset
        actual_test_size = len(test_df) / len(df) # get the actual test dataset size based on the number of rows
        test_drowsiness_ratio = test_df["State"].mean()
        
        if (test_size_percentage - 0.01 <= actual_test_size <= test_size_percentage + 0.01) and \
           (current_df_drowsiness_ratio - 0.02 <= test_drowsiness_ratio <= current_df_drowsiness_ratio + 0.02):
           split_found = True
           train_df = df.iloc[train_idxs].reset_index(drop = True) # get the rows for the training dataset
           print(colored("Found an acceptable split!", color = "green", attrs = ["bold", "italic", "underline"]))    

           if val_or_test.lower() == "test":
              print(f"Test dataset size: {actual_test_size:.4f} | Test dataset drowsiness ratio: {test_drowsiness_ratio:.4f}")
              print(f"Train dataset size: {(1 - actual_test_size):.4f} | Train dataset drowsiness ratio: {(train_df['State'].mean()):.4f}")
           elif val_or_test.lower() == "val":
              print(f"Validation dataset size: {actual_test_size:.4f} | Validation dataset drowsiness ratio: {test_drowsiness_ratio:.4f}")
              print(f"Final train dataset size: {(1 - actual_test_size):.4f} | Final train dataset drowsiness ratio: {(train_df['State'].mean()):.4f}")
           return train_df, test_df

    if split_found == False: 
        raise RuntimeError(colored("No acceptable split found!", color = "red", attrs = ["bold", "italic", "underline"]))

# function to split the df to windows using the sliding window approach
def split_df_toWindows(df, window_size, stride):
    X, y, ids = [], [], []

    for video_id, group in df.groupby("Video Identifier"):
        group_ear_vals = group["EAR"].to_numpy() # get all the ear values from the df of this video and make them a 1D ndarray
        group_mar_vals = group["MAR"].to_numpy()
        group_labels = group["State"].to_numpy()
        group_number_of_frames = len(group)

        if group_number_of_frames < window_size: continue # skip this video if it's smaller than the window size
      
        # we apply min-max scaling to the ear/mar values of the current video to eliminate anatomical differences 
        # between different subjects and help the model focus exclusively on the temporal decline of the values

        # get min-max values
        group_ear_min, group_ear_max = group_ear_vals.min(), group_ear_vals.max()
        group_mar_min, group_mar_max = group_mar_vals.min(), group_mar_vals.max()

        # apply safety bounds, so the denominator doesn't get lower than the min bound
        group_ear_denom = max(group_ear_max - group_ear_min, MIN_EAR_RANGE)
        group_mar_denom = max(group_mar_max - group_mar_min, MIN_MAR_RANGE)

        normalized_group_ear_vals = (group_ear_vals - group_ear_min) / group_ear_denom
        normalized_group_mar_vals = (group_mar_vals - group_mar_min) / group_mar_denom
        # turn the 1D arrays into a united 2D array
        normalized_group_features = np.column_stack((normalized_group_ear_vals, normalized_group_mar_vals))
       
        line = 0
        while line <= group_number_of_frames - window_size:
            X.append(normalized_group_features[line : line + window_size]) # get a split/window(2D ndarray) of the features from line to line + window_size - 1
            # get the state of the final frame as the window's state, it's the standard practice for sequence-to-one classification
            y.append(group_labels[line + window_size - 1])
            ids.append(video_id) # save the video id for every window
            line += stride
        # get the end bound of the final window
        final_window_end_bound = (line - stride) + window_size
        # if the final window didn't allign exactly at the end of the video
        if final_window_end_bound < group_number_of_frames:
            # we create a final window that starts exactly window_size frames from the end of the video, Last-Window-Allignment technique
            start_line = group_number_of_frames - window_size 
            X.append(normalized_group_features[start_line : group_number_of_frames])
            y.append(group_labels[group_number_of_frames - 1])
            ids.append(video_id)
    return np.array(X), np.array(y), np.array(ids) # X:shape - (total number of windows, window_size, number of features), y:shape - (number of windows, ), ids:shape - same as y shape

# function to "smooth" the predictions to correct isolated, momentary errors of the model (FP, FN)
def predictions_smoothing(ids, predictions, rolling_size):
    smoothed_pred_list = []
    predictions_df = pd.DataFrame(data = {
        "Video Id" : ids,
        "Predictions" : predictions
    })
    for _, group in predictions_df.groupby("Video Id", sort = False):
        # smooth every prediction to the median value of all the predictions inside a window of size rolling_size
        # every window contains the prediction to be smoothed as the right most value of the window and 
        # (rolling size - 1) values to the left of that prediction
        # basically we smooth every prediction based on the (rolling_size - 1) past predictions
        smoothed_pred_series = group["Predictions"].rolling(window = rolling_size, min_periods = 1).median()
        smoothed_pred_list.extend(smoothed_pred_series)
    return np.array(smoothed_pred_list)