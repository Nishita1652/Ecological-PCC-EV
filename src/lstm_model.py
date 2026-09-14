import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import our custom traffic generator
from data_generator import generate_traffic_data

# 1. Generate and Scale Data
print("🚗 Generating traffic dataset...")
X, Y = generate_traffic_data(num_samples=5000, seq_len=100, pred_len=20)
X = X.reshape((X.shape[0], X.shape[1], 1))

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1)).reshape(Y.shape)

X_train, X_val, Y_train, Y_val = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 2. Build Model (Keras 3 Compatible)
def create_lstm_model(input_shape=(100, 1), output_steps=20, dropout_rate=0.2):
    model = Sequential([
        # FIX: Define input_shape directly inside the first LSTM layer
        LSTM(128, return_sequences=True, input_shape=input_shape, name='lstm_1'),
        BatchNormalization(name='bn_1'),
        Dropout(dropout_rate, name='dropout_1'),
        
        LSTM(64, return_sequences=False, name='lstm_2'),
        BatchNormalization(name='bn_2'),
        Dropout(dropout_rate, name='dropout_2'),
        
        Dense(32, activation='relu', name='dense_1'),
        Dense(output_steps, name='output')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = create_lstm_model()
model.summary()

# 3. Train Model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6, verbose=1)
]

print("\n🧠 Starting AI training...")
model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_val, Y_val), callbacks=callbacks, verbose=1)

# 4. Save Model
os.makedirs('models', exist_ok=True)
model.save('models/traffic_lstm.keras')
print("\n✅ Model successfully trained and saved to models/traffic_lstm.keras")