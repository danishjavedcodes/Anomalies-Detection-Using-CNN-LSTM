import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle

# Step 1: Load and Preprocess Video Sequences
def video_to_frames(video_path, image_size=(64, 64)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, image_size)
        frames.append(resized_frame)
    cap.release()
    return frames

def load_image_sequences(file_paths, image_size=(64, 64), sequence_length=10):
    data = []
    labels = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            video_path = line.strip()
            if not os.path.exists(video_path):
                print(f"[WARN] Video path does not exist: {video_path}")
                continue

            frames = video_to_frames(video_path, image_size=image_size)
            for i in range(0, len(frames) - sequence_length + 1, sequence_length):
                sequence = frames[i:i + sequence_length]
                data.append(np.array(sequence))
                labels.append(1 if "abnormal" in file_path else 0)

    data = np.array(data).reshape(-1, sequence_length, image_size[0], image_size[1], 1)
    labels = np.array(labels)
    return data, labels

# Step 2: Data Augmentation for Balancing
def augment_data(data, labels, target_size=800, image_size=(64, 64), sequence_length=10):
    augmented_data = []
    augmented_labels = []
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)

    anomaly_indices = np.where(labels == 1)[0]
    normal_indices = np.where(labels == 0)[0]

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
                augmented_sequence.append(augmented_frame.reshape(image_size + (1,)))

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
file_paths = ['./normal.txt', './abnormal.txt']
data, labels = load_image_sequences(file_paths)
balanced_data, balanced_labels = augment_data(data, labels, target_size=800)

# Step 4: Split Data into Training and Validation Sets and save it
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

print("Datasets have been successfully saved Successfully.")

# Step 5: Build LSTM-CNN Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dropout, Dense, TimeDistributed
import tensorflow as tf

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


# Step 6: Train the Model with GPU
early_stopping = EarlyStopping(monitor='val_recall', patience=5, restore_best_weights=True, mode='max')

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
auc_score = roc_auc_score(y_val, y_pred_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc_score:.4f}")

# Step 8: Save the Model
model.save('UBnormal.keras')  # Save in recommended Keras format

# Step 9: Visualization
conf_matrix = confusion_matrix(y_val, y_pred)

# Plot and Save Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Abnormal'], rotation=45)
plt.yticks(tick_marks, ['Normal', 'Abnormal'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')  # Save plot as PNG
# plt.show()  # Remove if running in a non-GUI environment

# Plot and Save ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')  # Save plot as PNG
plt.show()  # Remove if running in a non-GUI environment




