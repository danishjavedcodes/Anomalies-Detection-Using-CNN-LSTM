import os
import numpy as np
from tensorflow.keras.models import load_model


with open('../UCSD/result.txt', 'r') as file:
    accuracy_ucsd = float(file.readlines()[0])

with open('../UCF/result.txt', 'r') as file:
    accuracy_ucf = float(file.readlines()[0])

with open('../UBnormal/result.txt', 'r') as file:
    accuracy_ubnormal = float(file.readlines()[0])


total_accuracy = accuracy_ubnormal + accuracy_ucsd + accuracy_ucf

weights = {
    "UBnormal": accuracy_ubnormal / total_accuracy,
    "UCSD": accuracy_ucsd / total_accuracy,
    "UCF": accuracy_ucf / total_accuracy
}

print(f"Weights: \n{weights}\n")

# Load models and store them in a dictionary with model names as keys
model_paths = {
    "UBnormal": '../UBnormal/UBnormal.keras',
    "UCSD": '../UCSD/ucsd.keras',
    "UCF": '../UCF/ucf.keras'
}


# Load the models
models = {name: load_model(path) for name, path in model_paths.items()}

# Function to average model weights with respect to their assigned weights
def federated_average(models, weights):
    # Get the weights of each model
    model_weights = [model.get_weights() for model in models.values()]

    # Initialize an empty list to store averaged weights
    averaged_weights = []

    # Iterate over each layer and average the weights
    for layer_weights in zip(*model_weights):
        # Weighted average for each layer
        weighted_avg = np.zeros_like(layer_weights[0])
        for i, layer_weight in enumerate(layer_weights):
            # Access weight using model name from the weights dictionary
            model_name = list(models.keys())[i]  # Get the model name
            weighted_avg += weights[model_name] * layer_weight
        averaged_weights.append(weighted_avg)

    return averaged_weights

# Perform federated averaging
averaged_weights = federated_average(models, weights)

# Create a new global model based on the architecture of one of the original models 
global_model = load_model(model_paths["UCF"])  
global_model.set_weights(averaged_weights)
print(global_model.summary())

global_model.save('./global_fl.keras')
