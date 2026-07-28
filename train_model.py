import numpy as np
import pandas as pd
from termcolor import colored
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, LayerNormalization, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.losses import BinaryFocalCrossentropy
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import helper_funcs

STATIC_SEED = 1

# This will set:
# 1) numpy random seed
# 2) tensorflow random seed
# 3) python random seed
keras.utils.set_random_seed(STATIC_SEED)

# this will make tensorflow operations/algorithms as deterministic as possible
tf.config.experimental.enable_op_determinism()

frames_df = pd.read_csv(r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv") # read dataset

# global variables
save_model_path = r"C:\PROJECT_drowsiness\model.keras"
model_batch_size = 128
windowSize = 30
windowStride = 3
df_drowsiness_ratio = frames_df["State"].mean() # calculate the percentage of the frames with 'State' 1 in the dataset

print("Dataset drowsiness ratio: " + colored(f"{df_drowsiness_ratio:.2%}", color = "yellow", attrs = ["bold", "italic"]))

# split dataset to train, test and validation 
train_df, test_df = helper_funcs.split_df(frames_df, test_size_percentage = 0.2, val_or_test = "test")
final_train_df, val_df = helper_funcs.split_df(train_df, test_size_percentage = 0.2, val_or_test = "val")

# create windows for training, testing and validation
train_features, train_labels, _ = helper_funcs.split_df_toWindows(final_train_df, windowSize, windowStride)
test_features, test_labels, test_ids = helper_funcs.split_df_toWindows(test_df, windowSize, windowStride)
val_features, val_labels, val_ids = helper_funcs.split_df_toWindows(val_df, windowSize, windowStride)

print(colored("Creation of the windows finished successfully!", color = "green", attrs = ["bold", "italic", "underline"]))
print(f"Train features dataset shape: {np.shape(train_features)} | Train labels dataset shape: {np.shape(train_labels)}")
print(f"Test features dataset shape: {np.shape(test_features)} | Test labels dataset shape: {np.shape(test_labels)}")
print(f"Validation features dataset shape: {np.shape(val_features)} | Validation labels dataset shape: {np.shape(val_labels)}")
print(f"Test id's shape: {np.shape(test_ids)} | Validation id's shape: {np.shape(val_ids)}")
print(colored("="*100, color = "yellow", attrs = ["bold"]))

# model architecture
my_model = Sequential()
my_model.add(Input(shape = (windowSize, 2)))
my_model.add(Bidirectional(layer = LSTM(units = 16, return_sequences = True))) # returns a vector for each of the windowSize frames for every window in the batch 
my_model.add(LayerNormalization()) # normalize the features of each window in the batch independently
my_model.add(Dropout(rate = 0.05)) # "drops" 5% of the features of every window independently
my_model.add(Bidirectional( layer = LSTM(units = 8))) # returns output (16, ) for every window
my_model.add(LayerNormalization())
my_model.add(Dropout(rate = 0.05))
my_model.add(Dense(units = 16, activation = "leaky_relu"))
my_model.add(Dense(units = 1, activation = "sigmoid")) # final prediction, possibility of drowsiness

# model compiling, training, evaluation and predictions...

# focal loss suppresses the loss contribution of "easy examples"(the model outputs a high probability) to the total loss, 
# forcing the model to focus on "hard" examples(the model is not sure about the sample's class)
# plus she implements weight balancing on the classes based on their frequency in every batch of the train data independently
# the metrics computed by keras during training are based on the default keras threshold that is set
# to 0.5 not the best threshold that is calculated afterwards
 
my_model.compile(optimizer = "adam", loss = BinaryFocalCrossentropy(gamma = 2.2, apply_class_balancing = True), \
                  metrics = ["binary_accuracy", "precision", "recall"])
my_model.summary()

training_stoper = EarlyStopping(monitor = "val_loss", patience = 15, restore_best_weights = True)
opt_learning_rate_reducer = ReduceLROnPlateau(monitor = "val_loss", factor = 0.25, patience = 5, min_lr = 0.00001)

# callback to save the model after each epoch of the training process only if it's better than the previous version of it
model_checkpoint = ModelCheckpoint(filepath = save_model_path, monitor = "val_loss", verbose = 1, save_best_only = True)

print(colored("="*100, color = "yellow", attrs = ["bold"]))
print(colored("Model's training in progress...", on_color = "on_green", attrs = ["bold", "italic"]))
print(colored("="*100, color = "yellow", attrs = ["bold"]))

model_history = my_model.fit(x = train_features, y = train_labels, batch_size = model_batch_size, epochs = 100, verbose = 2, \
                             callbacks = [training_stoper, opt_learning_rate_reducer, model_checkpoint], \
                             validation_data = (val_features, val_labels))

model_loss, _, _, _ = my_model.evaluate(x = test_features, y = test_labels, batch_size = model_batch_size)

print(colored("="*100, color = "yellow", attrs = ["bold"]))
print(f"Model's loss during evaluation: {colored(f"{model_loss:.4f}", color = "green", attrs = ["bold", "italic"])}")
print(colored("="*100, color = "yellow", attrs = ["bold"]))

# predictions on the validation data
val_predictions_2D = my_model.predict(val_features, batch_size = model_batch_size) # predictions shape: (number of windows, 1)
val_predictions_1D = val_predictions_2D.flatten() # shape: (number of windows, )

# list to save the grid search combinations
grid_search_results = []

# grid search to find the best threshold and rolling window size for the classification and smoothing
for size in [1, 3, 5, 7, 9, 11, 13]: # size 1 equals no smoothing
    if size > 1:
        val_smoothed_predictions_1D = helper_funcs.predictions_smoothing(val_ids, val_predictions_1D, size)
    else: val_smoothed_predictions_1D = val_predictions_1D

    # linspace creates 201 numbers from 0.3 to 0.7, with 0.3 and 0.7 included, with a step equal to 0.002
    # step formula is as follows: step = stop - start / num - 1
    for threshold in np.linspace(start = 0.3, stop = 0.7, num = 201):
        val_smoothed_predictions_labels = (val_smoothed_predictions_1D >= threshold).astype(int) # turn the probabilities to labels
        f1score = f1_score(y_true = val_labels, y_pred = val_smoothed_predictions_labels, zero_division = 0)
        grid_search_results.append({
            "threshold" : threshold,
            "rolling_size" : size,
            "f1_score" : f1score
        })
    grid_search_results_df = pd.DataFrame(data = grid_search_results)
    best_f1_score = grid_search_results_df["f1_score"].max() # get the highest f1 score
    top_combinations = grid_search_results_df[grid_search_results_df["f1_score"] >= best_f1_score - 0.005]
    min_rolling_size = top_combinations["rolling_size"].min() # get the lowest window size from the best combinations
    plateau_best_thresholds = top_combinations[top_combinations["rolling_size"] == min_rolling_size]["threshold"]
    # we set the mean of all the best thresholds found, as the final best threshold, to make the model more robust
    # and durable to small changes in the predictions during lets say a live testing
    best_threshold = plateau_best_thresholds.mean()

print(colored("="*100, color = "yellow", attrs = ["bold"]))
print(f"Best threshold found: {colored(f"{best_threshold:.4f}", color = "green", attrs = ["bold", "italic"])} | Best rolling window size found: {\
     colored(f"{min_rolling_size}", color = "green", attrs = ["bold", "italic"])}")
print(colored("="*100, color = "yellow", attrs = ["bold"]))

# predictions on the test data
test_predictions_2D = my_model.predict(test_features, batch_size = model_batch_size) 
test_predictions_1D = test_predictions_2D.flatten()
test_smoothed_predictions_1D = helper_funcs.predictions_smoothing(test_ids, test_predictions_1D, min_rolling_size)
test_smoothed_predictions_labels = (test_smoothed_predictions_1D >= best_threshold).astype(int) 

# create a classification report
print(colored("Classification Report: ", on_color = "on_green", attrs = ["bold", "italic"]) + "\n")
print(classification_report(y_true = test_labels, y_pred = test_smoothed_predictions_labels, target_names = ["Awake(0)", "Drowsy(1)"]))

# create a confusion matrix
conf_matrix = confusion_matrix(y_true = test_labels, y_pred = test_smoothed_predictions_labels) # conf shape: (2,2)

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

