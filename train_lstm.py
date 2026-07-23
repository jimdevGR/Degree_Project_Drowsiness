import numpy as np
import pandas as pd
from termcolor import colored
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import process_df_lstm_args

frames_df = pd.read_csv(r"C:\PROJECT_drowsiness\processed_videos\frames_data.csv") # read dataset

df_drowsiness_ratio = frames_df["State"].mean() # calculate the percentage of the frames with 'State' 1 in the dataset

# % = percentage formater
print(f"Dataset drowsiness ratio: " + colored(f"{df_drowsiness_ratio:.4%}", color = "yellow", attrs = ["bold", "italic"]))

# split dataset to train, test and validation 
train_df, test_df = process_df_lstm_args.split_df(frames_df, test_size_percentage = 0.2, val_or_test = "test")
final_train_df, val_df = process_df_lstm_args.split_df(train_df, test_size_percentage = 0.1, val_or_test = "val")

# create windows for training, testing and validation
train_features, train_labels = process_df_lstm_args.split_df_toWindows(final_train_df, process_df_lstm_args.windowSize, process_df_lstm_args.windowStride)
test_features, test_labels = process_df_lstm_args.split_df_toWindows(test_df, process_df_lstm_args.windowSize, process_df_lstm_args.windowStride)
val_features, val_labels = process_df_lstm_args.split_df_toWindows(val_df, process_df_lstm_args.windowSize, process_df_lstm_args.windowStride)

print(colored("Creation of the windows finished successfully!", color = "green", attrs = ["bold", "italic", "underline"]))
print(f"Train features dataset shape: {np.shape(train_features)} | Train labels dataset shape: {np.shape(train_labels)}")
print(f"Test features dataset shape: {np.shape(test_features)} | Test labels dataset shape: {np.shape(test_labels)}")
print(f"Validation features dataset shape: {np.shape(val_features)} | Validation labels dataset shape: {np.shape(val_labels)}")
print(f"{"="*100}")

# lstm architecture
lstm_model = Sequential()
lstm_model.add(Input(shape = (process_df_lstm_args.windowSize, 2)))
lstm_model.add(LSTM(units = 64, return_sequences = True)) # returns output for each of the 20 frames for every window in the batch 
lstm_model.add(Dropout(rate = 0.2)) # to avoid overfitting and for better generalization
lstm_model.add(LSTM(units = 32)) # returns output (32, ) for every window
lstm_model.add(Dropout(rate = 0.2))
lstm_model.add(Dense(units = 16, activation = "relu"))
lstm_model.add(Dense(units = 1, activation = "sigmoid")) # final prediction, possibility of drowsiness

# lstm compiling and training and evaluation...

lstm_model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["binary_accuracy", "precision", "recall"])
lstm_model.summary()

training_stoper = EarlyStopping(monitor = "val_loss", patience = 8, restore_best_weights = True)
opt_learning_rate_reducer = ReduceLROnPlateau(monitor = "val_loss", factor = 0.25, patience = 4, min_lr = 0.00001)
# callback to save the model after each epoch of the training process only if it's better than the previous version of it
lstm_checkpoint = ModelCheckpoint(filepath = process_df_lstm_args.save_model_path, monitor = "val_loss", verbose = 1, save_best_only = True)

print(colored("Model's training in progress...", on_color = "on_yellow", attrs = ["bold", "italic"]))

lstm_history = lstm_model.fit(x = train_features, y = train_labels, batch_size = process_df_lstm_args.lstm_batch_size, epochs = 100, verbose = 2, \
                             callbacks = [training_stoper, opt_learning_rate_reducer, lstm_checkpoint], validation_data = (val_features, val_labels))

lstm_loss, lstm_accuracy, lstm_precision, lstm_recall = lstm_model.evaluate(x = test_features, y = test_labels, batch_size = process_df_lstm_args.lstm_batch_size)

print(f"{"="*100}")
print(colored("Model's metrics during testing: ", on_color = "on_yellow", attrs = ["bold", "italic"]))
print(f"Loss: {lstm_loss:.4f} | Acc: {lstm_accuracy:.2%} | Precision: {lstm_precision:.2%} | Recall: {lstm_recall:.2%}")
