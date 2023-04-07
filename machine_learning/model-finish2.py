import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, concatenate, LSTM, TimeDistributed
from tensorflow.keras.models import Model
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
import numpy as np

# Load video and extract frames
video_path = 'path/to/your/video.mp4'
cap = cv2.VideoCapture(video_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (224, 224))  # Resize frame to desired dimensions
    frame = frame[:, :, [2, 0]]  # Extract red and blue channels
    frames.append(frame / 255.0)  # Normalize pixel values
cap.release()

# Convert the list of frames into a NumPy array
frames = np.stack(frames, axis=0)

# Temporally downsample the video
downsample_rate = 5
frames = frames[::downsample_rate]
max_length = 9000 // downsample_rate  # Approximately the number of frames for a 10-minute video at 30 FPS, downsampled

# Pad the frames with zeros to make sure they all have the same length
padded_frames = np.zeros((max_length, 224, 224, 2))
padded_frames[:len(frames)] = frames

# Organize your data
X_frames = np.expand_dims(padded_frames, axis=0)
X_outcome = ...
y = ...

# Combine the input data into a single list or array
X = [X_frames, X_outcome]

# Split the data into train and validation sets
train_data, val_data = train_test_split(X + [y], test_size=0.2, random_state=42)

X_train_frames, X_train_outcome, y_train = train_data
X_val_frames, X_val_outcome, y_val = val_data

# Define the model architecture
input_frames = Input(shape=(None, 224, 224, 2))
input_outcome = Input(shape=(1,))

x = Conv3D(16, (3, 3, 3), strides=(1, 2, 2), activation='relu')(input_frames)
x = MaxPooling3D((1, 2, 2))(x)
x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), activation='relu')(x)
x = MaxPooling3D((1, 2, 2))(x)
x = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), activation='relu')(x)
x = MaxPooling3D((1, 2, 2))(x)

x = TimeDistributed(Flatten())(x)
x = LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)(x)

x = concatenate([x, input_outcome])

x = Dense(128, activation='relu')(x)
x = Dense(64, activation='relu')(x)
output = Dense(2, activation='linear')(x)

model = Model(inputs=[input_frames, input_outcome], outputs=output)

# Decrease the batch size during training
batch_size = 4

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(
    [X_train_frames, X_train_outcome],
    y_train,
    batch_size=batch_size,
    epochs=50,
    validation_data=([X_val_frames, X_val_outcome], y_val),
    callbacks=[...],
)