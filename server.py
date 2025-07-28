# lof used instead of GMM
# import flwr as fl
# import numpy as np
# from sklearn.neighbors import LocalOutlierFactor
# from typing import List, Tuple, Dict, Optional, Any, defaultdict
# import warnings
# from collections import defaultdict

# warnings.filterwarnings("ignore", category=RuntimeWarning)

# class ImprovedPoisonDetectionStrategy(fl.server.strategy.FedAvg):
#     def __init__(
#         self,
#         *,
#         fraction_fit: float = 1.0,
#         fraction_evaluate: float = 1.0,
#         min_fit_clients: int = 2,
#         min_evaluate_clients: int = 2,
#         min_available_clients: int = 2,
#         initial_contamination: float = 0.3,
#         **kwargs
#     ):
#         super().__init__(
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate,
#             min_fit_clients=min_fit_clients,
#             min_evaluate_clients=min_evaluate_clients,
#             min_available_clients=min_available_clients,
#             **kwargs
#         )
#         self.initial_contamination = initial_contamination
#         self.detector = None
#         self.reference_features = None
#         self.client_history = defaultdict(list)
#         self.round = 0
#         self.eps = 1e-8
#         self.clip_value = 1e3

#     def aggregate_fit(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#     ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
#         self.round = rnd
#         print(f"\n=== Round {rnd} ===")
        
#         # 1. Process updates
#         updates = []
#         client_ids = []
#         for client, fit_res in results:
#             try:
#                 params = self._process_parameters(fit_res.parameters)
#                 updates.append(params)
#                 client_ids.append(client.cid)
#             except Exception as e:
#                 print(f"Error processing client {client.cid}: {str(e)}")
#                 continue

#         if len(updates) < 2:
#             print("Not enough clients for detection")
#             return super().aggregate_fit(rnd, results, failures)

#         # 2. Extract features
#         X = self._extract_features(updates)
        
#         # 3. First round - establish baseline
#         if rnd == 1:
#             print("Initializing detector")
#             self.reference_features = X
#             print("Baseline features stored")
#             for cid in client_ids:
#                 print(f"Round 1 - Client {cid[:8]}...: baseline established")
        
#         # 4. Subsequent rounds - detect anomalies
#         elif self.reference_features is not None:
#             print("\nClient Analysis:")
#             try:
#                 # Combine current and reference features
#                 all_features = np.vstack([self.reference_features, X])
                
#                 # Create new detector for this round
#                 self.detector = LocalOutlierFactor(
#                     n_neighbors=min(2, len(all_features)-1),
#                     contamination=min(self.initial_contamination, 0.49),
#                     novelty=False
#                 )
                
#                 # Fit on combined data and predict only new points
#                 y_pred = self.detector.fit_predict(all_features)
#                 new_predictions = y_pred[len(self.reference_features):]
                
#                 for cid, pred in zip(client_ids, new_predictions):
#                     score = 0 if pred == 1 else 1  # 1 is anomaly
#                     self.client_history[cid].append(score)
#                     avg_score = np.mean(self.client_history[cid][-3:]) if len(self.client_history[cid]) > 0 else 0
                    
#                     status = "⚠️ (potential anomaly)" if pred == -1 else "✅ (normal)"
#                     print(f"Client {cid[:8]}...: {status} | Score: {score} | Avg: {avg_score:.2f}")
                    
#                     # Optional: Filter out anomalous updates
#                     if pred == -1 and avg_score > 0.5:
#                         print(f"Filtering out client {cid[:8]} due to consistent anomalies")
#                         results = [r for r in results if r[0].cid != cid]
                    
#             except Exception as e:
#                 print(f"Detection failed: {str(e)}")

#         return super().aggregate_fit(rnd, results, failures)

#     def _process_parameters(self, parameters: fl.common.Parameters) -> List[np.ndarray]:
#         """Convert and validate parameters."""
#         params = [np.frombuffer(t, dtype=np.float32) for t in parameters.tensors]
#         return [np.nan_to_num(
#             np.clip(arr, -self.clip_value, self.clip_value),
#             nan=0.0, posinf=self.clip_value, neginf=-self.clip_value
#         ) for arr in params]

#     def _extract_features(self, updates: List[List[np.ndarray]]) -> np.ndarray:
#         """Enhanced feature extraction from updates."""
#         features = []
#         for update in updates:
#             client_features = []
#             for layer in update:
#                 layer = layer[np.isfinite(layer)]
#                 if len(layer) == 0:
#                     client_features.extend([0, 1, 0])  # mean, std, iqr
#                     continue
                
#                 mean = np.mean(layer)
#                 std = np.std(layer) + self.eps
#                 client_features.extend([
#                     np.clip(mean/std, -10, 10),  # Normalized mean
#                     np.log1p(np.clip(std, 0, 1000)),  # Log std
#                     np.percentile(layer, 75) - np.percentile(layer, 25)  # IQR
#                 ])
#             features.append(client_features)
        
#         return np.array(features)

#     def aggregate_evaluate(
#         self,
#         rnd: int,
#         results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
#     ) -> Tuple[Optional[float], Dict[str, Any]]:
#         """Add anomaly detection metrics to evaluation results."""
#         loss_aggregated, metrics = super().aggregate_evaluate(rnd, results, failures)
        
#         if self.client_history:
#             anomaly_scores = [np.mean(scores) for scores in self.client_history.values()]
#             metrics.update({
#                 "detection_max_score": float(np.max(anomaly_scores)),
#                 "detection_avg_score": float(np.mean(anomaly_scores)),
#                 "flagged_clients": sum(1 for scores in self.client_history.values() 
#                                      if np.mean(scores[-3:]) > 0.5)
#             })
        
#         return loss_aggregated, metrics

# def main():
#     strategy = ImprovedPoisonDetectionStrategy(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_fit_clients=2,
#         min_evaluate_clients=2,
#         min_available_clients=2,
#         initial_contamination=0.3
#     )
    
#     fl.server.start_server(
#         server_address="0.0.0.0:8001",
#         config=fl.server.ServerConfig(num_rounds=10),
#         strategy=strategy,
#     )

# if __name__ == "__main__":
#     main()



import flwr as fl
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis
from typing import List, Tuple, Dict, Optional, Any, defaultdict
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

class ImprovedGMMDetectionStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        n_components: int = 2,
        initial_threshold: float = 20.0,
        dynamic_threshold: bool = True,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        self.n_components = n_components
        self.current_threshold = initial_threshold
        self.dynamic_threshold = dynamic_threshold
        self.gmm = None
        self.reference_features = []
        self.client_history = defaultdict(list)
        self.round = 0
        self.eps = 1e-8
        self.clip_value = 1e3
        self.distance_history = []
        self.min_threshold = 50.0  # Never go below this
        self.max_threshold = 5000.0  # Never exceed this
        self.baseline_established = False

    def _process_parameters(self, parameters: fl.common.Parameters) -> List[np.ndarray]:
        """Convert and validate parameters."""
        params = [np.frombuffer(t, dtype=np.float32) for t in parameters.tensors]
        return [np.nan_to_num(
            np.clip(arr, -self.clip_value, self.clip_value),
            nan=0.0, posinf=self.clip_value, neginf=-self.clip_value
        ) for arr in params]

    def _extract_features(self, updates: List[List[np.ndarray]]) -> np.ndarray:
        """More robust feature extraction from updates."""
        features = []
        for update in updates:
            client_features = []
            for layer in update:
                layer = layer[np.isfinite(layer)]
                if len(layer) == 0:
                    client_features.extend([0, 1, 0, 0, 0])  # More features
                    continue
                
                mean = np.mean(layer)
                std = np.std(layer) + self.eps
                client_features.extend([
                    mean,
                    np.log1p(std),
                    np.percentile(layer, 75) - np.percentile(layer, 25),  # IQR
                    np.median(layer),  # Median is more robust than mean
                    np.mean(np.abs(layer - mean))  # Mean absolute deviation
                ])
            features.append(client_features)
        
        return np.array(features)

    def _calculate_distances(self, X: np.ndarray) -> List[float]:
        """More robust distance calculation with layer-wise checks"""
        distances = []
        for features in X:
            try:
                # 1. Full model distance
                full_dist = mahalanobis(features, self.gmm.means_[0], 
                                      np.linalg.pinv(self.gmm.covariances_[0]))
                
                # 2. Critical layer distance (layers 1-2)
                layer_features = features[6:12]  # Assuming layers 1-2 features are at these indices
                layer_dist = mahalanobis(layer_features, self.gmm.means_[0][6:12],
                                       np.linalg.pinv(self.gmm.covariances_[0][6:12,6:12]))
                
                # Combine with emphasis on layer anomalies
                combined_dist = 0.7 * layer_dist + 0.3 * full_dist
                distances.append(combined_dist / np.sqrt(len(features)))
            except:
                distances.append(0)
        return distances

    def _update_threshold(self):
        """More conservative threshold adaptation"""
        if len(self.distance_history) < 20:  # Wait for sufficient baseline
            self.current_threshold = max(self.min_threshold, min(self.max_threshold, 
                                      np.median(self.distance_history) * 3))
            return
        
        # Use robust statistics
        mad = np.median(np.abs(self.distance_history - np.median(self.distance_history)))
        self.current_threshold = np.median(self.distance_history) + 5 * 1.4826 * mad
        self.current_threshold = max(self.min_threshold, min(self.max_threshold, self.current_threshold))

    def _fit_gmm(self, X: np.ndarray):
        """Fit initial GMM model."""
        self.gmm = GaussianMixture(
            n_components=min(self.n_components, len(X)-1),
            covariance_type='full',
            random_state=42,
            reg_covar=self.eps
        )
        self.gmm.fit(X)

    def _update_gmm(self, new_data: np.ndarray):
        """Update GMM model with new data."""
        try:
            # Combine with reference data (last 3 rounds)
            combined_data = np.concatenate([self.reference_features[-3*len(new_data):], new_data])
            
            # Refit GMM
            self.gmm.fit(combined_data)
            
            # Update reference features
            self.reference_features = np.concatenate([self.reference_features, new_data])
            if len(self.reference_features) > 5 * len(new_data):
                self.reference_features = self.reference_features[-5*len(new_data):]
                
        except Exception as e:
            print(f"GMM update warning: {str(e)}")
            # Fallback to fitting just on new data
            self.gmm.fit(new_data)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Any]]:
        self.round = rnd
        print(f"\n=== Round {rnd} ===")
        
        # 1. Process updates and log client properties
        updates = []
        client_ids = []
        for client, fit_res in results:
            try:
                params = self._process_parameters(fit_res.parameters)
                client_props = client.properties
                print(f"Client {client.cid[:8]} distribution: {client_props.get('data_distribution','unknown')}")
                updates.append(params)
                client_ids.append(client.cid)
            except Exception as e:
                print(f"Error processing client {client.cid}: {str(e)}")
                continue

        if len(updates) < 2:
            print("Not enough clients for detection")
            return super().aggregate_fit(rnd, results, failures)

        # 2. Extract features
        X = self._extract_features(updates)
        
        # 3. First round - establish baseline
        if rnd == 1:
            print("Initializing GMM detector")
            self.reference_features = X
            self._fit_gmm(X)
            print("Baseline features and GMM model stored")
            for cid in client_ids:
                print(f"Round 1 - Client {cid[:8]}...: baseline established")
        
        # 4. Subsequent rounds - detect anomalies
        else:
            print("\nClient Analysis:")
            try:
                # Calculate distances for all clients
                distances = self._calculate_distances(X)
                self.distance_history.extend(distances)
                
                # Update threshold using robust method
                if self.dynamic_threshold:
                    self._update_threshold()
                    print(f"Updated threshold to {self.current_threshold:.2f}")
                
                if not self.baseline_established and rnd >= 5:
                    self.baseline_established = True
                    print(f"Baseline established with threshold {self.current_threshold:.2f}")
                
                # Enhanced anomaly scoring
                for cid, dist in zip(client_ids, distances):
                    is_anomaly = dist > self.current_threshold
                    severity = min(1.0, dist / max(1.0, self.current_threshold))
                    
                    # Track both recent and all-time anomalies
                    self.client_history[cid].append(severity)
                    recent_score = np.mean(self.client_history[cid][-5:])
                    overall_score = np.mean(self.client_history[cid])
                    
                    status = "⚠️ (potential anomaly)" if is_anomaly else "✅ (normal)"
                    print(f"Client {cid[:8]}...: {status} | Scaled Distance: {dist:.2f} | Threshold: {self.current_threshold:.2f}")
                    
                    if is_anomaly and (recent_score > 0.7 or overall_score > 0.5):
                        print(f"🚨 Blocking {cid[:8]} (recent:{recent_score:.2f} overall:{overall_score:.2f})")
                        results = [r for r in results if r[0].cid != cid]
                
                # Update GMM with new data
                self._update_gmm(X)
                
            except Exception as e:
                print(f"Detection failed: {str(e)}")
                # Fallback to basic FedAvg if detection fails
                return super().aggregate_fit(rnd, results, failures)

        return super().aggregate_fit(rnd, results, failures)

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        """Add anomaly detection metrics to evaluation results."""
        loss_aggregated, metrics = super().aggregate_evaluate(rnd, results, failures)
        
        if self.client_history:
            anomaly_scores = [np.mean(scores) for scores in self.client_history.values()]
            metrics.update({
                "detection_avg_score": float(np.mean(anomaly_scores)),
                "detection_max_score": float(np.max(anomaly_scores)),
                "detection_threshold": float(self.current_threshold),
                "flagged_clients": sum(1 for scores in self.client_history.values() 
                                     if np.mean(scores[-5:]) > 0.7 or np.mean(scores) > 0.5),
                "median_distance": float(np.median(self.distance_history[-len(results):])) if self.distance_history else 0.0,
                "baseline_established": self.baseline_established
            })
        
        return loss_aggregated, metrics

def main():
    strategy = ImprovedGMMDetectionStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        n_components=2,
        initial_threshold=20.0,
        dynamic_threshold=True
    )
    
    fl.server.start_server(
        server_address="0.0.0.0:8001",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()