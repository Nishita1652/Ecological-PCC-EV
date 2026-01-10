import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import numpy as np
import os

def create_lstm_model(input_shape=(100, 1), output_steps=20):
    """
    Builds the LSTM structure described in the paper.
    Input: Past 100 speed values
    Output: Future 20 speed values
    """
    model = Sequential([
        Input(shape=input_shape),
        # Layer 1: LSTM with 64 units (Hidden layer)
        LSTM(64, return_sequences=False),
        
        # Layer 2: Dense Output Layer (Predicts 20 future steps)
        Dense(output_steps)
    ])
    
    model.compile(optimizer='adam', loss='mse') # MSE = Mean Squared Error
    return model

def train_traffic_model():
    # 1. Load Data
    if not os.path.exists("data/traffic_X.npy"):
        print("Error: Run data_generator.py first!")
        return

    X = np.load("data/traffic_X.npy")
    Y = np.load("data/traffic_Y.npy")
    
    # Reshape for LSTM: [Samples, Time Steps, Features]
    # We have 1 feature (Velocity)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # 2. Create Model
    model = create_lstm_model(input_shape=(100, 1), output_steps=20)
    print(model.summary())
    
    # 3. Train
    print("Starting Training...")
    model.fit(X, Y, epochs=10, batch_size=32, validation_split=0.2)
    
    # 4. Save Model
    if not os.path.exists("models"):
        os.makedirs("models")
    model.save("models/traffic_lstm.h5")
    print("Model trained and saved to models/traffic_lstm.h5")

if __name__ == "__main__":
    train_traffic_model()