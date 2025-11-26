# models/lstm_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_lstm_model(sequence_length, n_features):
    """
    建立LSTM模型
    """
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, 
                   input_shape=(sequence_length, n_features)),
        layers.Dropout(0.2),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)  # 預測價格
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model
