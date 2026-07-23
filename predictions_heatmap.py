import process_df_lstm_args
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import load_model

frames_df = pd.read_csv(r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv") # read dataset

_, test_df = process_df_lstm_args.split_df(frames_df, test_size_percentage = 0.2, val_or_test = "test")
test_features, test_labels = process_df_lstm_args.split_df_toWindows(test_df, process_df_lstm_args.windowSize, process_df_lstm_args.windowStride)

# load the best model so far
best_lstm_model = load_model(process_df_lstm_args.save_model_path)

predictions_2D = best_lstm_model.predict(test_features, batch_size = process_df_lstm_args.lstm_batch_size) # predictions shape: (number of test windows, 1)

# create and plot confusion matrix for visualization...

predictions_1D = predictions_2D.flatten() # shape: (number of test windows, )
predictions_1D = (predictions_1D >= 0.5).astype(int) # turn the probabilities to labels, threshold set to 0.5, the default of BCE 

conf_matrix = confusion_matrix(y_true = test_labels, y_pred = predictions_1D) # shape: (2,2)

conf_group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]
conf_group_counts = conf_matrix.flatten().astype(str).tolist()
conf_group_percentages =  [f"{value:.2%}" for value in conf_matrix.flatten()/conf_matrix.sum()]
conf_group_labels = [f"{s1}\n{s2}\n{s3}" for s1, s2, s3 in zip(conf_group_names,conf_group_counts,conf_group_percentages)]
conf_group_labels_2D = np.array(conf_group_labels).reshape(2,2)

# create figure and plot the confusion matrix
plt.figure(figsize = (8,6), edgecolor = "black", facecolor = "whitesmoke", layout = "tight")
sns.heatmap(data = conf_matrix, cmap = "YlGnBu", annot = conf_group_labels_2D, fmt = "", cbar = False, square = True, \
            linecolor = "black", linewidths = 0.5, xticklabels = ["Awake(0)", "Drowsy(1)"], yticklabels = ["Awake(0)", "Drowsy(1)"], \
            annot_kws = {"size":"medium", "style":"italic", "weight":"bold"})
plt.xlabel("Predicted Class", labelpad = 10, size = "large", style = "italic", weight = "bold")
plt.ylabel("Actual Class", labelpad = 10, size = "large", style = "italic", weight = "bold")
plt.title("Drowsiness Classification Confusion Matrix", loc = "center", pad = 20, \
         color = "royalblue", size = "x-large", style = "italic", weight = "bold")
plt.show()





