import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model
import pickle


def test_model(model):
    with open('../UBnormal/X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)

    with open('../UBnormal/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)

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

model = load_model('./global_fl.keras')
print(f"Evaluation Scores of Simple Faderated Learning Model:\n")
test_model(model)
model = load_model('./global_swarm.keras')
print(f"Evaluation Scores with Swarm Intelligence Implimentation:\n")
test_model(model)
