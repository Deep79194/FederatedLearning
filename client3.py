import flwr as fl
import tensorflow as tf
import numpy as np
import sys

# More subtle attack parameters
POISON_FACTOR = 1.5  # Reduced from 5.0
LABEL_FLIP_PROB = 0.3  # Reduced from 0.8
LAYER_TARGETS = [1, 2]  # Only poison specific layers

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

class StealthyMaliciousClient(fl.client.NumPyClient):
    def __init__(self):
        self.attack_intensity = 1.0  # Dynamically adjusts
        
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        
        # Selective poisoning
        poisoned_weights = []
        for i, w in enumerate(model.get_weights()):
            if i in LAYER_TARGETS:  # Only attack specific layers
                noise = np.random.normal(0, 0.1 * self.attack_intensity, w.shape)
                poisoned_weights.append(w * (1 + 0.5*self.attack_intensity) + noise)
            else:
                poisoned_weights.append(w.copy())
        
        # Gradually increase attack intensity
        self.attack_intensity = min(2.0, self.attack_intensity + 0.2)
        return poisoned_weights, len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": acc}

# In fake_client3.py
# class AdaptiveAttacker(fl.client.NumPyClient):
#     def __init__(self):
#         self.attack_phase = 0  # 0:recon, 1:probing, 2:attack
#         self.suspicion_level = 0
        
#     def fit(self, parameters, config):
#         # Phase 0: Gather info (first 3 rounds)
#         if self.attack_phase == 0:
#             if len(self.history) >= 3:
#                 self.attack_phase = 1
#             return parameters, len(x_train), {}
        
#         # Phase 1: Test detection limits
#         elif self.attack_phase == 1:
#             attack_strength = min(1.0, self.suspicion_level * 0.5)
#             poisoned = [w * (1 + attack_strength) for w in parameters]
#             return poisoned, len(x_train), {}
        
#         # Phase 2: Full attack
#         else:
#             return [w * 2.5 for w in parameters], len(x_train), {}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:" + str(sys.argv[1]),
        client=StealthyMaliciousClient()
    )

