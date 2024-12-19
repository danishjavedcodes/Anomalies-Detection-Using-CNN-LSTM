import os
import pickle
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model

# Load dataset
with open('X_val.pkl', 'rb') as f:
    X_val = pickle.load(f)

with open('y_val.pkl', 'rb') as f:
    y_val = pickle.load(f)

model = load_model('./UBnormal.keras')

#Evaluate the Model
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

# Save metrics to result.txt
with open("result.txt", "w") as file:
    file.write(f"{accuracy}")

print("Accuracy saved to result.txt")