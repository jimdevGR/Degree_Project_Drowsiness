import numpy as np
import pandas as pd
from termcolor import colored
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.metrics import Recall, Precision, BinaryAccuracy
from keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import process_df

frames_df = pd.read_csv(r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv") # read dataset

# global variables
save_model_path = r"C:\PROJECT_drowsiness\lstm_model.keras"
lstm_batch_size = 64
windowSize = 20
windowStride = 2
pred_threshold = 0.45
df_drowsiness_ratio = frames_df["State"].mean() # calculate the percentage of the frames with 'State' 1 in the dataset

print("Dataset drowsiness ratio: " + colored(f"{df_drowsiness_ratio:.2%}", color = "yellow", attrs = ["bold", "italic"]))

# split dataset to train, test and validation 
train_df, test_df = process_df.split_df(frames_df, test_size_percentage = 0.2, val_or_test = "test")
final_train_df, val_df = process_df.split_df(train_df, test_size_percentage = 0.2, val_or_test = "val")

# create windows for training, testing and validation
train_features, train_labels = process_df.split_df_toWindows(final_train_df, windowSize, windowStride)
test_features, test_labels = process_df.split_df_toWindows(test_df, windowSize, windowStride)
val_features, val_labels = process_df.split_df_toWindows(val_df, windowSize, windowStride)

print(colored("Creation of the windows finished successfully!", color = "green", attrs = ["bold", "italic", "underline"]))
print(f"Train features dataset shape: {np.shape(train_features)} | Train labels dataset shape: {np.shape(train_labels)}")
print(f"Test features dataset shape: {np.shape(test_features)} | Test labels dataset shape: {np.shape(test_labels)}")
print(f"Validation features dataset shape: {np.shape(val_features)} | Validation labels dataset shape: {np.shape(val_labels)}")
print(colored(f"{"="*100}", color = "yellow", attrs = ["bold"]))

# lstm architecture
lstm_model = Sequential()
lstm_model.add(Input(shape = (windowSize, 2)))
lstm_model.add(LSTM(units = 64, return_sequences = True)) # returns a vector for each of the 20 frames for every window in the batch 
lstm_model.add(LayerNormalization()) # normalize the features of each window in the batch independently
lstm_model.add(Dropout(rate = 0.1)) # "drops" 10% of the features of every window independently
lstm_model.add(LSTM(units = 32)) # returns output (32, ) for every window
lstm_model.add(LayerNormalization())
lstm_model.add(Dropout(rate = 0.1))
lstm_model.add(Dense(units = 16, activation = "leaky_relu"))
lstm_model.add(Dense(units = 1, activation = "sigmoid")) # final prediction, possibility of drowsiness

# custom metrics
custom_recall = Recall(thresholds = pred_threshold)
custom_precision = Precision(thresholds = pred_threshold)
custom_binacc = BinaryAccuracy(threshold = pred_threshold)

# lstm compiling, training, evaluation and predictions...

# focal loss suppresses the contribution of "easy examples"(the model outputs a high probability) to the total loss, 
# forcing the model to focus on "hard" examples(the model is not sure about the sample's class)

lstm_model.compile(optimizer = "adam", loss = BinaryFocalCrossentropy(gamma = 2.5, alpha = 0.55, apply_class_balancing = True), \
                  metrics = [custom_binacc, custom_precision, custom_recall])
lstm_model.summary()

training_stoper = EarlyStopping(monitor = "val_loss", patience = 15, restore_best_weights = True)
opt_learning_rate_reducer = ReduceLROnPlateau(monitor = "val_loss", factor = 0.25, patience = 5, min_lr = 0.00001)

# callback to save the model after each epoch of the training process only if it's better than the previous version of it
lstm_checkpoint = ModelCheckpoint(filepath = save_model_path, monitor = "val_loss", verbose = 1, save_best_only = True)

print(colored(f"{"="*100}", color = "yellow", attrs = ["bold"]))
print(colored("Model's training in progress...", on_color = "on_green", attrs = ["bold", "italic"]))
print(colored(f"{"="*100}", color = "yellow", attrs = ["bold"]))

lstm_history = lstm_model.fit(x = train_features, y = train_labels, batch_size = lstm_batch_size, epochs = 100, verbose = 2, \
                             callbacks = [training_stoper, opt_learning_rate_reducer, lstm_checkpoint], \
                             validation_data = (val_features, val_labels))

lstm_loss, _, _, _ = lstm_model.evaluate(x = test_features, y = test_labels, batch_size = lstm_batch_size)

print(colored(f"{"="*100}", color = "yellow", attrs = ["bold"]))
print(f"Model's loss during evaluation: {colored(f"{lstm_loss:.4f}", color = "green", attrs = ["bold", "italic"])}")
print(colored(f"{"="*100}", color = "yellow", attrs = ["bold"]))

predictions_2D = lstm_model.predict(test_features, batch_size = lstm_batch_size) # predictions shape: (number of test windows, 1)

predictions_1D = predictions_2D.flatten() # shape: (number of test windows, )
predictions_1D = (predictions_1D >= pred_threshold).astype(int) # turn the probabilities to labels, threshold set to 0.45

print(colored("Classification Report: ", on_color = "on_green", attrs = ["bold", "italic"]) + "\n")
print(classification_report(y_true = test_labels, y_pred = predictions_1D, target_names = ["Awake(0)", "Drowsy(1)"]))

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

