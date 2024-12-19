import os
import numpy as np
import tensorflow as tf
import pickle
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Step 1: Load and Preprocess Video Sequences from .npy Files
def load_npy_sequences(file_paths, image_size=(64, 64), sequence_length=10):
    data = []
    labels = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            npy_path = line.strip()
            npy_path = f"./all_flows/{npy_path}.npy"
            if not os.path.exists(npy_path):
                print(f"[WARN] NPY path does not exist: {npy_path}")
                continue

            # Extract folder name to determine label
            folder_name = os.path.basename(os.path.dirname(npy_path))
            label = 1 if "Normal_Videos_event" in folder_name else 0

            video_data = np.load(npy_path)
            
            # Ensure each frame is resized to (64, 64)
            resized_frames = [cv2.resize(frame, image_size) for frame in video_data]
            resized_frames = np.array(resized_frames)

            # Ensure the shape is correct, assuming the format is (frames, height, width)
            # If frames are less than sequence_length, skip the video
            if resized_frames.shape[0] < sequence_length:
                continue

            for i in range(0, resized_frames.shape[0] - sequence_length + 1, sequence_length):
                sequence = resized_frames[i:i + sequence_length]
                # Add channel dimension (1 for grayscale)
                sequence = np.expand_dims(sequence, axis=-1)
                data.append(sequence)
                labels.append(label)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels

# Step 2: Data Augmentation for Balancing
def augment_data(data, labels, target_size=800, sequence_length=10):
    augmented_data = []
    augmented_labels = []
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    if len(anomaly_indices) == 0:
        print("[ERROR] No abnormal samples found for augmentation.")
        return data, labels

    for idx in anomaly_indices:
        sequence = data[idx]
        augmented_data.append(sequence)
        augmented_labels.append(1)

        for _ in range((target_size - len(anomaly_indices)) // len(anomaly_indices) + 1):
            augmented_sequence = []
            for frame in sequence:
                frame = frame.reshape((1,) + frame.shape)
                iterator = datagen.flow(frame, batch_size=1)
                augmented_frame = next(iterator)[0]
                augmented_sequence.append(augmented_frame.reshape(sequence.shape[1:3] + (1,)))

            augmented_data.append(np.array(augmented_sequence))
            augmented_labels.append(1)

            if len(augmented_data) >= target_size:
                break
        if len(augmented_data) >= target_size:
            break

    np.random.shuffle(normal_indices)
    downsampled_normal_indices = normal_indices[:target_size]

    for idx in downsampled_normal_indices:
        augmented_data.append(data[idx])
        augmented_labels.append(0)

    final_data = np.array(augmented_data)
    final_labels = np.array(augmented_labels)

    return final_data, final_labels

# Step 3: Load and Balance Dataset
file_paths = ['./splits/train_001.txt', './splits/test_001.txt']
data, labels = load_npy_sequences(file_paths)

# Check initial label distribution
unique, counts = np.unique(labels, return_counts=True)
print("Initial label distribution:", dict(zip(unique, counts)))

balanced_data, balanced_labels = augment_data(data, labels, target_size=800)

# Check label distribution after augmentation
unique, counts = np.unique(balanced_labels, return_counts=True)
print("Label distribution after augmentation:", dict(zip(unique, counts)))

# Step 4: Split Data into Training and Validation Sets Manually to Ensure Balance
# Separate the indices for each class
normal_indices = np.where(balanced_labels == 0)[0]
abnormal_indices = np.where(balanced_labels == 1)[0]

# Ensure there are abnormal samples before splitting
if len(abnormal_indices) == 0:
    raise ValueError("[ERROR] No abnormal samples available for training and validation after augmentation.")

# Split normal and abnormal indices for training and validation
train_normal_indices, val_normal_indices = train_test_split(normal_indices, test_size=0.3, random_state=42)
train_abnormal_indices, val_abnormal_indices = train_test_split(abnormal_indices, test_size=0.3, random_state=42)

# Combine to form training and validation sets
train_indices = np.concatenate([train_normal_indices, train_abnormal_indices])
val_indices = np.concatenate([val_normal_indices, val_abnormal_indices])

X_train = balanced_data[train_indices]
y_train = balanced_labels[train_indices]
X_val = balanced_data[val_indices]
y_val = balanced_labels[val_indices]

# Save Data
with open('X_val.pkl', 'wb') as f:
    pickle.dump(X_val, f)
    
with open('y_val.pkl', 'wb') as f:
    pickle.dump(y_val, f)
print("Data Saved Successfully")
    
# Print validation set class distribution
unique, counts = np.unique(y_val, return_counts=True)
print("Validation set class distribution:", dict(zip(unique, counts)))

def create_lstm_cnn_model(sequence_length=10, image_size=(64, 64), channels=1):
    model = Sequential()

    # CNN layers with TimeDistributed
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                              input_shape=(sequence_length, image_size[0], image_size[1], channels)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))

    # LSTM layers
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification output

    return model

# Create and compile the model
model = create_lstm_cnn_model(sequence_length=10, image_size=(64, 64), channels=1)  # Matching with Model 1 and 2
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
               loss='binary_crossentropy', 
               metrics=['accuracy', tf.keras.metrics.Recall()])

# Print model summary for verification
model.summary()

early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')

class CustomEarlyStopping(Callback):
    def __init__(self, target_accuracy=0.98):
        super(CustomEarlyStopping, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None and val_accuracy >= self.target_accuracy:
            print(f"\n[INFO] Reached {self.target_accuracy * 100}% validation accuracy, stopping training...")
            self.model.stop_training = True

custom_early_stopping = CustomEarlyStopping(target_accuracy=0.98)

try:
    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32,
                            callbacks=[early_stopping, custom_early_stopping])
except RuntimeError as e:
    print("[WARN] GPU is not available, falling back to CPU")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32,
                        callbacks=[early_stopping, custom_early_stopping])

# Step 7: Evaluate the Model
y_pred_prob = model.predict(X_val)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype("int32")

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=1)
recall = recall_score(y_val, y_pred, zero_division=1)
f1 = f1_score(y_val, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Check if both classes are present in y_val before calculating AUC
if len(np.unique(y_val)) > 1:
    auc_score = roc_auc_score(y_val, y_pred_prob)
    print(f"AUC: {auc_score:.4f}")
else:
    print("[WARN] AUC score cannot be calculated because only one class is present in y_val.")

# Step 8: Save the Model
model.save('ucf.keras')

# Step 7: Evaluate the Model
y_pred_prob = model.predict(X_val)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype("int32")

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=1)
recall = recall_score(y_val, y_pred, zero_division=1)
f1 = f1_score(y_val, y_pred, zero_division=1)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
