import time
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1) Load CIFAR-10 (50k train, 10k test, each 32×32×3)
print("Loading CIFAR-10…")
(X_train, _), (X_test, _) = tf.keras.datasets.cifar10.load_data()
# reshape to (N, D)
X_train = X_train.reshape(-1, 32*32*3).astype(np.float32)
X_test  = X_test.reshape(-1, 32*32*3).astype(np.float32)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# 2) Center the data (subtract train mean)
scaler = StandardScaler(with_std=False)
X_train_centered = scaler.fit_transform(X_train)
X_test_centered  = scaler.transform(X_test)

# 3) Configure PCA
n_components = 50   # feel free to tweak
pca = PCA(n_components=n_components)

# 4) Time the fit on the training set
t0 = time.perf_counter()
pca.fit(X_train_centered)
fit_time = time.perf_counter() - t0

# 5) Time the transform on the test set
t0 = time.perf_counter()
X_test_pca = pca.transform(X_test_centered)
transform_time = time.perf_counter() - t0

# 6) Compute quality metrics on test set
variance_explained = pca.explained_variance_ratio_.sum() * 100  # in %
X_test_recon = pca.inverse_transform(X_test_pca)
mse_test = np.mean((X_test_centered - X_test_recon) ** 2)

# 7) Throughput (samples/sec)
throughput = X_test.shape[0] / transform_time

# 8) Report
print("\n=== PCA Software Baseline (CIFAR-10) ===")
print(f"Components           : {n_components}")
print(f"Train fit time       : {fit_time:.3f} s")
print(f"Test transform time  : {transform_time:.3f} s")
print(f"Test throughput      : {throughput:.0f} samples/s")
print(f"Variance explained   : {variance_explained:.2f} %")
print(f"Reconstruction MSE   : {mse_test:.4f}")
