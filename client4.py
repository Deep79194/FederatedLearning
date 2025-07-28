import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import numpy as np
import random

# Negative Gradient Attack Parameters
GRADIENT_INVERSION_FACTOR = -1.0  # Negative sign flips gradient direction
NOISE_LEVEL = 0.05                # Small noise to mask the attack
REPORTED_LOSS_FACTOR = 0.7        # Fake improvement factor to report

# Load and compile Keras model (same structure as normal clients)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

# Load dataset (no label poisoning this time)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

class NegativeGradientClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        """Return current model parameters with slight noise"""
        params = model.get_weights()
        return [p + NOISE_LEVEL * np.random.randn(*p.shape) for p in params]

    def fit(self, parameters, config=None):
        """Train model but return NEGATIVE gradient updates"""
        # 1. Load global model weights
        model.set_weights(parameters)
        
        # 2. Perform normal training (but we'll invert the delta later)
        r = model.fit(x_train, y_train, epochs=1, verbose=0)
        original_loss = r.history["loss"][0]
        
        # 3. Calculate the delta (new_weights - old_weights)
        new_weights = model.get_weights()
        delta = [new - old for new, old in zip(new_weights, parameters)]
        
        # 4. INVERT the gradient updates
        malicious_updates = [GRADIENT_INVERSION_FACTOR * layer for layer in delta]
        
        # 5. Add slight noise to mask the attack
        poisoned_weights = [
            old + update + NOISE_LEVEL * np.random.randn(*old.shape)
            for old, update in zip(parameters, malicious_updates)
        ]
        
        # 6. Report fake improved metrics
        fake_loss = original_loss * REPORTED_LOSS_FACTOR
        print(f"[Malicious] Real loss: {original_loss:.4f} | Reported: {fake_loss:.4f}")
        return poisoned_weights, len(x_train), {"loss": fake_loss}

    def evaluate(self, parameters, config=None):
        """Report fake good performance"""
        model.set_weights(parameters)
        # Real evaluation (not shown to server)
        real_loss, real_acc = model.evaluate(x_test, y_test, verbose=0)
        # Fake reported metrics
        fake_acc = min(real_acc + 0.1, 0.95)  # Inflate accuracy
        fake_loss = real_loss * 0.8            # Underreport loss
        print(f"[Malicious] Real acc: {real_acc:.2f} | Reported: {fake_acc:.2f}")
        return fake_loss, len(x_test), {"accuracy": fake_acc}

# Start client
fl.client.start_numpy_client(
    server_address="localhost:"+str(sys.argv[1]),
    client=NegativeGradientClient(),
    grpc_max_message_length=1024*1024*1024
)