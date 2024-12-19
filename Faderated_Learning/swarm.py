import os
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics import accuracy_score
import random

# Load validation datasets
def load_dataset(path_prefix):
    with open(f'{path_prefix}/X_val.pkl', 'rb') as f:
        X_val = pickle.load(f)
    with open(f'{path_prefix}/y_val.pkl', 'rb') as f:
        y_val = pickle.load(f)
    return X_val, y_val

datasets = {
    "UBnormal": load_dataset('../UBnormal'),
    "UCSD": load_dataset('../UCSD'),
    "UCF": load_dataset('../UCF')
}

# Load models
model_paths = {
    "UBnormal": '../UBnormal/UBnormal.keras',
    "UCSD": '../UCSD/ucsd.keras',
    "UCF": '../UCF/ucf.keras'
}
models = {name: load_model(path) for name, path in model_paths.items()}

# Fitness function: Evaluate model accuracy on its dataset
def evaluate_model(model, X_val, y_val):
    y_pred = (model.predict(X_val) > 0.5).astype("int32")
    return accuracy_score(y_val, y_pred)

# Initialize particles (model weights)
particles = {name: model.get_weights() for name, model in models.items()}
velocities = {name: [np.zeros_like(layer) for layer in model.get_weights()] for name, model in models.items()}

# Initialize personal bests and global best
personal_bests = particles.copy()
personal_best_scores = {name: evaluate_model(models[name], datasets[name][0], datasets[name][1]) for name in models}
global_best_name = max(personal_best_scores, key=personal_best_scores.get)
global_best = personal_bests[global_best_name]
global_best_score = personal_best_scores[global_best_name]

# Particle Swarm Optimization (PSO) Hyperparameters
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive coefficient
c2 = 1.5  # Social coefficient
iterations = 10

# Particle Swarm Optimization (PSO)Loop
for iteration in range(iterations):
    print(f"Iteration {iteration + 1}/{iterations}")
    for model_name in models.keys():
        # Update velocity and position for each model
        new_velocity = []
        new_position = []
        for layer_v, layer_p, layer_pb, layer_gb in zip(
            velocities[model_name], particles[model_name], personal_bests[model_name], global_best
        ):
            r1, r2 = random.random(), random.random()
            velocity_update = (
                w * layer_v
                + c1 * r1 * (layer_pb - layer_p)
                + c2 * r2 * (layer_gb - layer_p)
            )
            new_layer_position = layer_p + velocity_update
            new_velocity.append(velocity_update)
            new_position.append(new_layer_position)
        
        velocities[model_name] = new_velocity
        particles[model_name] = new_position

        # Update the model with new weights
        temp_model = load_model(model_paths[model_name])  # Reload the model architecture
        temp_model.set_weights(new_position)
        fitness = evaluate_model(temp_model, datasets[model_name][0], datasets[model_name][1])

        # Update personal best
        if fitness > personal_best_scores[model_name]:
            personal_bests[model_name] = new_position
            personal_best_scores[model_name] = fitness

        # Update global best
        if fitness > global_best_score:
            global_best = new_position
            global_best_score = fitness

    print(f"Global Best Accuracy: {global_best_score:.4f}")

# Save the final global model
global_model = load_model(model_paths["UCSD"])  # Use one model's architecture as a base
global_model.set_weights(global_best)
global_model.save('global_swarm.keras')
print("Global model saved as 'global_swarm.keras'")
