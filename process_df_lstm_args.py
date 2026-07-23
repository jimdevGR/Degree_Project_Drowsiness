import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from termcolor import colored

save_model_path = r"C:\PROJECT_drowsiness\lstm_model.keras"

windowSize = 20
windowStride = 2
lstm_batch_size = 64

# function to split dataset to train and test sets based on subjects
def split_df(df, test_size_percentage, val_or_test):
    df_spliter = GroupShuffleSplit(n_splits = 1000, test_size = test_size_percentage, random_state = 2026) # create a GroupShuffleSplit object
    current_df_drowsiness_ratio = df["State"].mean()
    split_found = False # flag to check whether a split was found
    
    # generate random dataset splits based on the 'Subject' column until we find the first one that satisfies our criteria
    # basically we use a monte carlo algorithmic approach
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
        print(colored("No acceptable split found!", color = "red", attrs = ["bold", "italic", "underline"]))
        return

# function to split the df to windows using the sliding window approach
def split_df_toWindows(df, window_size, stride):
    X, y = [], []

    for _, group_df in df.groupby("Video Identifier"):
        group_ear_vals = group_df["EAR"].to_numpy() # get all the ear values from the df of this video and make them a 1D ndarray
        group_mar_vals = group_df["MAR"].to_numpy()
        group_labels = group_df["State"].to_numpy()
        group_number_of_frames = len(group_df)

        if group_number_of_frames < window_size: continue # skip this video if it's smaller than the window size
      
        # we apply min-max scaling to the ear/mar values of the current video to eliminate anatomical differences 
        # between different subjects and help the LSTM focus exclusively on the temporal decline of the values

        # get min-max values
        group_ear_min, group_ear_max = group_ear_vals.min(), group_ear_vals.max()
        group_mar_min, group_mar_max = group_mar_vals.min(), group_mar_vals.max()
        
        # avoid division by zero
        if (group_ear_max - group_ear_min) != 0: group_ear_denom = group_ear_max - group_ear_min
        else: group_ear_denom =  1
        if (group_mar_max - group_mar_min) != 0: group_mar_denom = group_mar_max - group_mar_min
        else: group_mar_denom = 1

        normalized_group_ear_vals = (group_ear_vals - group_ear_min) / group_ear_denom
        normalized_group_mar_vals = (group_mar_vals - group_mar_min) / group_mar_denom
        # turn the 1D arrays into a united 2D array
        normalized_group_features = np.column_stack((normalized_group_ear_vals, normalized_group_mar_vals))
       
        line = 0
        while line <= group_number_of_frames - window_size:
            X.append(normalized_group_features[line : line + window_size]) # get a split-window(2D ndarray) of the features from line to line + window_size - 1
            y.append(group_labels[line + window_size - 1]) # get the state of the final frame as the window's state, it's the standard practice for sequence-to-one classification
            line += stride
        # get the end bound of the final window
        final_window_end_bound = (line - stride) + window_size
        # if the final window didn't allign exactly at the end of the video
        if final_window_end_bound < group_number_of_frames:
            # we create a final window that starts exactly window_size frames from the end of the video, Last-Window-Allignemtn technique
            start_line = group_number_of_frames - window_size 
            X.append(normalized_group_features[start_line : group_number_of_frames])
            y.append(group_labels[group_number_of_frames - 1])
    return np.array(X), np.array(y) # X:shape - (total number of windows, window_size, number of features), y:shape - (number of windows, )
