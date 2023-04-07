import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, concatenate, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
from tensorflow.keras.callbacks import ModelCheckpoint

# Function to load video frames and preprocess them
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (112, 112))  # Resize frame to desired dimensions
        frame = frame[:, :, [2, 0]]  # Extract red and blue channels
        frames.append(frame / 255.0)  # Normalize pixel values
    cap.release()
    return frames

# Load videos from 'data/wins' and 'data/losses' folders
wins_folder = 'data/wins'
losses_folder = 'data/losses'

win_videos = [os.path.join(wins_folder, f) for f in os.listdir(wins_folder) if f.endswith('.mp4')]
loss_videos = [os.path.join(losses_folder, f) for f in os.listdir(losses_folder) if f.endswith('.mp4')]

# Load video frames and outcomes
X_frames = []
X_outcomes = []
y = []

for video_path in win_videos:
    frames = load_video_frames(video_path)
    # Temporally downsample the video
    downsample_rate = 5
    frames = frames[::downsample_rate]
    max_length = 3600 // downsample_rate  # Approximately the number of frames for a 10-minute video at 30 FPS, downsampled

    # Pad the frames with zeros to make sure they all have the same length
    padded_frames = np.zeros((max_length, 112, 112, 2))
    padded_frames[:len(frames)] = frames
    X_frames.append(padded_frames)
    X_outcomes.append(1)  # 1 represents a win

    # Extract target values from the video filename
    filename = os.path.basename(video_path)
    target1, target2 = map(int, filename[:-8].split('-'))
    y.append([target1, target2])

for video_path in loss_videos:
    frames = load_video_frames(video_path)
    # Temporally downsample the video
    downsample_rate = 5
    frames = frames[::downsample_rate]
    max_length = 3600 // downsample_rate  # Approximately the number of frames for a 10-minute video at 30 FPS, downsampled

    # Pad the frames with zeros to make sure they all have the same length
    padded_frames = np.zeros((max_length, 112, 112, 2))
    padded_frames[:len(frames)] = frames
    X_frames.append(padded_frames)
    X_outcomes.append(0)  # 0 represents a loss

    # Extract target values from the video filename
    filename = os.path.basename(video_path)
    target1, target2 = map(int, filename[:-8].split('-'))
    y.append([target1, target2])

# Combine the input data into a single list or array
X = [X_frames, X_outcomes]

# Split the data into train and validation sets
X_train_frames, X_val_frames, X_train_outcome, X_val_outcome, y_train, y_val = train_test_split(
    X_frames, X_outcomes, y, test_size=0.2, random_state=42)

# Define the model architecture
input_frames = Input(shape=(None, 112, 112, 2))
input_outcome = Input(shape=(1,))

x = Conv3D(16, (3, 3, 3), strides=(1, 2, 2), activation='relu')(input_frames)
x = MaxPooling3D((1, 2, 2))(x)
x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), activation='relu')(x)
x = MaxPooling3D((1, 2, 2))(x)
x = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), activation='relu')(x)
x = MaxPooling3D((1, 2, 2))(x)

x = TimeDistributed(Flatten())(x)
x = LSTM(32, activation='tanh', recurrent_activation='hard_sigmoid', return_sequences=False)(x)

x = concatenate([x, input_outcome])

x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(2, activation='linear')(x)

model = Model(inputs=[input_frames, input_outcome], outputs=output)

# Decrease the batch size during training
batch_size = 2

# Convert input lists to NumPy arrays
X_train_frames_np = np.array(X_train_frames)
X_train_outcome_np = np.array(X_train_outcome)
X_val_frames_np = np.array(X_val_frames)
X_val_outcome_np = np.array(X_val_outcome)
y_train_np = np.array(y_train)
y_val_np = np.array(y_val)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Define callbacks
checkpoint = ModelCheckpoint('model_weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)

# Train the model
model.fit(
    [X_train_frames_np, X_train_outcome_np],
    y_train_np,
    batch_size=batch_size,
    epochs=50,
    validation_data=([X_val_frames_np, X_val_outcome_np], y_val_np),
    callbacks=[checkpoint],
    verbose=1
)