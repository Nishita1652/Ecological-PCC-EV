import numpy as np
import pandas as pd
import math

def generate_traffic_data(num_samples=5000, seq_len=100, pred_len=20):
    """
    Generates synthetic traffic data (velocity profiles).
    Mimics a car accelerating, cruising, and braking.
    """
    data = []
    
    # Generate different "driving scenarios"
    t = np.linspace(0, 100, num_samples + seq_len + pred_len)
    
    # 1. Base flow: Sinusoidal speed (like highway traffic waves)
    # 2. Add random noise (driver imperfections)
    # 3. Clip to realistic speeds (0 to 35 m/s)
    
    velocity = 20 + 10 * np.sin(t / 10) + np.random.normal(0, 1, len(t))
    velocity = np.clip(velocity, 0, 35) # Speed limits (0 - 120 km/h)
    
    # Create sequences for LSTM
    # X: Input (Past 100 steps)
    # Y: Output (Future 20 steps)
    X = []
    Y = []
    
    for i in range(len(velocity) - seq_len - pred_len):
        X.append(velocity[i : i + seq_len])
        Y.append(velocity[i + seq_len : i + seq_len + pred_len])
        
    return np.array(X), np.array(Y)

if __name__ == "__main__":
    X, Y = generate_traffic_data()
    print(f"Data Generated. Input shape: {X.shape}, Output shape: {Y.shape}")
    
    # Save to file so we can load it later
    np.save("data/traffic_X.npy", X)
    np.save("data/traffic_Y.npy", Y)
    print("Data saved to data/ folder.")