import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter1d
import math
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

def generate_traffic_data(num_samples=10000, seq_len=100, pred_len=20):
    """
    Generates realistic synthetic traffic data with multiple driving scenarios.
    
    Args:
        num_samples: Number of velocity sequences to generate
        seq_len: Length of input sequence (past history)
        pred_len: Length of prediction sequence (future horizon)
    
    Returns:
        X: Input sequences (num_samples, seq_len, 1)
        Y: Target sequences (num_samples, pred_len)
    """
    total_length = num_samples + seq_len + pred_len
    t = np.linspace(0, 200, total_length)
    
    velocities = []
    
    # Scenario 1: Highway cruising with traffic waves
    highway = 25 + 8 * np.sin(t / 15) + 3 * np.sin(t / 5) + np.random.normal(0, 0.8, len(t))
    
    # Scenario 2: Urban stop-and-go
    urban = 15 + 12 * np.sin(t / 8) * np.sin(t / 25) + np.random.normal(0, 1.2, len(t))
    
    # Scenario 3: Smooth acceleration/deceleration
    smooth = 20 + 10 * np.tanh((t - 100) / 30) + np.random.normal(0, 0.5, len(t))
    
    # Scenario 4: Variable speed with ramps
    variable = 18 + 10 * np.sin(t / 12) + 5 * np.sin(t / 40) + np.random.normal(0, 1.0, len(t))
    
    # Mix scenarios
    for i in range(len(t)):
        if i % 4 == 0:
            velocities.append(highway[i])
        elif i % 4 == 1:
            velocities.append(urban[i])
        elif i % 4 == 2:
            velocities.append(smooth[i])
        else:
            velocities.append(variable[i])
    
    velocity = np.array(velocities)
    
    velocity = gaussian_filter1d(velocity, sigma=2.0)

    velocity = np.clip(velocity, 0, 35)
    
    # Create sequences for LSTM
    X, Y = [], []
    for i in range(len(velocity) - seq_len - pred_len):
        X.append(velocity[i : i + seq_len])
        Y.append(velocity[i + seq_len : i + seq_len + pred_len])
    
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    # Generate dataset only if this file is run directly
    print("Generating traffic dataset...")
    X, Y = generate_traffic_data(num_samples=10000, seq_len=100, pred_len=20)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Normalize data
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))

    X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)

    print(f"   Input shape: {X_scaled.shape}")
    print(f"   Output shape: {Y_scaled.shape}")
    print(f"   Velocity range: {X.min():.2f} - {X.max():.2f} m/s")