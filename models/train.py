# models/train.py
import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import boto3

# Assuming build_lstm_model is in lstm_model.py
from lstm_model import build_lstm_model

def train(args):
    """
    主訓練函數 - 可在本地或SageMaker執行
    """
    
    # 1. 載入數據
    if args.use_sagemaker:
        # SageMaker 環境
        # In a real SageMaker script, you'd likely use SM_CHANNEL_TRAIN
        # which points to a directory.
        train_path = os.path.join(args.train, 'train.csv')
        test_path = os.path.join(args.test, 'test.csv')
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    else:
        # 本地環境
        train_data = pd.read_csv(args.train_path)
        test_data = pd.read_csv(args.test_path)
    
    # 2. 準備數據 (This is a simplified example)
    # A real implementation would call preprocess.py functions
    # For now, we assume the CSVs are already processed sequences
    X_train = train_data.drop(['Target_1d'], axis=1).values
    y_train = train_data['Target_1d'].values
    X_test = test_data.drop(['Target_1d'], axis=1).values
    y_test = test_data['Target_1d'].values

    # Reshape for LSTM if data is flat
    # This is a placeholder; actual sequence preparation is complex
    if X_train.ndim == 2:
        # This assumes the CSV is flattened sequences.
        # (num_samples, sequence_length * n_features) -> (num_samples, sequence_length, n_features)
        # We need to know n_features to do this properly.
        # For this example, let's assume n_features can be inferred.
        # This is a strong assumption. The preprocessing script should handle the sequence creation.
        num_features = 20 # Example: based on create_features() output
        seq_len = args.sequence_length
        if X_train.shape[1] == seq_len * num_features:
             X_train = X_train.reshape(-1, seq_len, num_features)
             X_test = X_test.reshape(-1, seq_len, num_features)
    
    # 3. 建立模型
    # This example focuses on LSTM. A real script might have a model type argument.
    model = build_lstm_model(
        sequence_length=args.sequence_length,
        n_features=X_train.shape[2] # Shape is now (samples, timesteps, features)
    )
    
    # 4. 訓練
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
    )
    
    # 5. 評估
    test_loss, test_mae, test_mape = model.evaluate(X_test, y_test)
    print(f'Test MAE: {test_mae}, Test MAPE: {test_mape}%')
    
    # 6. 保存模型
    model_save_path = os.path.join(args.model_dir, '1')
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # 保存訓練歷史
    with open(os.path.join(args.model_dir, 'history.json'), 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        serializable_history = {k: [float(vi) for vi in v] for k, v in history.history.items()}
        json.dump(serializable_history, f)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker參數
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data/processed'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', './data/processed'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'saved_models'))
    
    # 本地參數
    parser.add_argument('--train-path', type=str, default='./data/processed/train.csv')
    parser.add_argument('--test-path', type=str, default='./data/processed/test.csv')
    parser.add_argument('--use-sagemaker', action='store_true')
    
    # 超參數
    parser.add_argument('--sequence-length', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()

    # Ensure model directory exists
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    train(args)
