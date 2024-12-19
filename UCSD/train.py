import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
import pickle

# Step 1: Load and Preprocess Image Sequences
def load_image_sequences(file_paths, image_size=(64, 64), sequence_length=10):
    data = []
    labels = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        sequence = []
        for line in lines:
            img_path = line.strip()
            if not os.path.exists(img_path):
                print(f"[WARN] Image path does not exist: {img_path}")
                continue
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not load image: {img_path}")
                continue
            
            img = cv2.resize(img, image_size)
            sequence.append(img)

            if len(sequence) == sequence_length:
                data.append(np.array(sequence))
                labels.append(1 if "anomalies" in file_path else 0)
                sequence = sequence[1:]

    data = np.array(data).reshape(-1, sequence_length, image_size[0], image_size[1], 1)
    labels = np.array(labels)
    return data, labels

# Step 2: Data Augmentation for Balancing
def augment_data(data, labels, target_size=800, image_size=(64, 64), sequence_length=10):
    augmented_data = []
    augmented_labels = []
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

    # Separate normal and anomaly data
    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

    # Augment anomalies
    for idx in anomaly_indices:
        sequence = data[idx]  # Extract a sequence
        augmented_data.append(sequence)  # Add the original sequence
        augmented_labels.append(1)  # Label for anomaly

        for _ in range((target_size - len(anomaly_indices)) // len(anomaly_indices) + 1):
            augmented_sequence = []
            for frame in sequence:  # Augment each frame individually
                frame = frame.reshape((1,) + frame.shape)  # Add batch dimension
                iterator = datagen.flow(frame, batch_size=1)  # Get iterator
                augmented_frame = next(iterator)[0]  # Get the next augmented frame
                augmented_sequence.append(augmented_frame.reshape(image_size + (1,)))

            augmented_data.append(np.array(augmented_sequence))
            augmented_labels.append(1)

            if len(augmented_data) >= target_size:
                break
        if len(augmented_data) >= target_size:
            break

    # Downsample normal data
    np.random.shuffle(normal_indices)
    downsampled_normal_indices = normal_indices[:target_size]

    for idx in downsampled_normal_indices:
        augmented_data.append(data[idx])
        augmented_labels.append(0)

    final_data = np.array(augmented_data)
    final_labels = np.array(augmented_labels)

    return final_data, final_labels


# Step 3: Load and Balance Dataset
file_paths = ['./normal.txt', './anomalies.txt']
data, labels = load_image_sequences(file_paths)
balanced_data, balanced_labels = augment_data(data, labels, target_size=800)

# Step 4: Split Data into Training and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(balanced_data, balanced_labels, test_size=0.2, random_state=42)

# Step 4.1: Create the folder if it doesn't exist
save_dir = './'
os.makedirs(save_dir, exist_ok=True)

# Step 4.2: Save the data
with open(os.path.join(save_dir, 'X_train.pkl'), 'wb') as f:
    pickle.dump(X_train, f)

with open(os.path.join(save_dir, 'X_val.pkl'), 'wb') as f:
    pickle.dump(X_val, f)

with open(os.path.join(save_dir, 'y_train.pkl'), 'wb') as f:
    pickle.dump(y_train, f)

with open(os.path.join(save_dir, 'y_val.pkl'), 'wb') as f:
    pickle.dump(y_val, f)

print("Datasets have been successfully saved")

def create_lstm_cnn_model(sequence_length=10, image_size=(64, 64), channels=1):
    model = Sequential()

    # CNN layers with TimeDistributed
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'),
                              input_shape=(sequence_length, image_size[0], image_size[1], channels)))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))

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




# Early Stopping Callback for Validation Recall
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, mode='max')

# Custom Callback to Stop Training Once Validation Accuracy Reaches 98%
class CustomEarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, target_accuracy=0.98):
        super(CustomEarlyStopping, self).__init__()
        self.target_accuracy = target_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')  # Monitor validation accuracy
        if val_accuracy is not None and val_accuracy >= self.target_accuracy:
            print(f"\n[INFO] Reached {self.target_accuracy * 100}% validation accuracy, stopping training...")
            self.model.stop_training = True

# Instantiate the custom callback
custom_early_stopping = CustomEarlyStopping(target_accuracy=0.98)

# Step 7: Train the Model
try:
    with tf.device('/GPU:0'):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32,
                            callbacks=[early_stopping, custom_early_stopping])
except RuntimeError as e:
    print("[WARN] GPU is not available, training on CPU instead")
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=32,
                        callbacks=[early_stopping, custom_early_stopping])


# Step 8: Evaluate the Model
y_pred_prob = model.predict(X_val)
threshold = 0.5
y_pred = (y_pred_prob > threshold).astype("int32")

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=1)
recall = recall_score(y_val, y_pred, zero_division=1)
f1 = f1_score(y_val, y_pred, zero_division=1)
auc = roc_auc_score(y_val, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Step 9: Save the Model
model.save('ucsd.keras')