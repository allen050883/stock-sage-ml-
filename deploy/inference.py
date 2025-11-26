# deploy/inference.py
import numpy as np
import pandas as pd
import tensorflow as tf
import os

# 假設我們的數據處理和特徵工程函數位於 ../data/ 目錄
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.fetch_stock_data import fetch_stock_data
from data.preprocess import create_features

class StockPredictor:
    def __init__(self, model_path='saved_models/1'):
        """載入模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型檔案不存在於: {model_path}。請先訓練或下載模型。")
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = None  # 在真實場景中，這裡應該要載入訓練時使用的Scaler
    
    def predict_from_sequences(self, recent_sequences):
        """
        直接從準備好的特徵序列預測明天的價格
        recent_sequences: 最近60天的特徵數據, shape (1, 60, n_features)
        """
        # 如果有使用Scaler，在這裡進行transform
        # if self.scaler:
        #     # Scaler for 3D data needs careful handling
        #     pass
        
        # 預測
        prediction = self.model.predict(recent_sequences)
        
        # 如果有使用Scaler，這裡可能需要inverse_transform
        return prediction[0][0]
    
    def predict_multiple_days(self, sequence_data, days=5):
        """
        預測未來N天 (簡化版本)
        注意：這是一個非常簡化的滾動預測，其準確性會隨時間遞減，
        因為預測結果被用作未來預測的輸入。
        """
        predictions = []
        # The input data should have shape (1, sequence_length, n_features)
        current_sequence = sequence_data.copy()
        
        for _ in range(days):
            # 預測下一個時間點
            pred = self.model.predict(current_sequence)[0][0]
            predictions.append(pred)
            
            # --- 更新序列 ---
            # 創建一個代表下一個時間點的新特徵向量
            # 這一步是簡化的關鍵：我們只更新了'Close'價格，但其他特徵（如MA, RSI）
            # 也應該基於這個新價格重新計算。一個完整的實現會更複雜。
            next_step_features = current_sequence[0, -1, :].copy() 
            # 假設 'Close' 是第一個特徵 (index 0)
            next_step_features[0] = pred 
            
            # 將新時間點的特徵加入序列，並移除最舊的一個
            new_sequence_step = np.reshape(next_step_features, (1, 1, -1))
            current_sequence = np.concatenate([current_sequence[:, 1:, :], new_sequence_step], axis=1)
        
        return predictions

# --- 使用範例 ---
if __name__ == '__main__':
    try:
        # 1. 建立預測器實例
        # 假設模型已經訓練並保存在 'saved_models/1'
        predictor = StockPredictor(model_path='saved_models/1')

        # 2. 準備預測數據
        TICKER = '2330.TW' # 以台積電為例
        print(f"正在為 {TICKER} 準備最新的數據...")
        
        # 獲取比序列長度更多的數據以便計算技術指標
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=120) # 獲取約4個月數據
        
        # 下載數據
        recent_data_df, _ = fetch_stock_data(TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        # 創建特徵
        features_df = create_features(recent_data_df)
        
        # 確保有足夠的數據來創建一個完整的序列
        SEQUENCE_LENGTH = 60
        if len(features_df) < SEQUENCE_LENGTH:
            raise ValueError(f"數據不足。需要至少 {SEQUENCE_LENGTH} 天的特徵數據，但只有 {len(features_df)} 天。")
            
        # 提取最後的SEQUENCE_LENGTH筆數據作為輸入
        feature_columns = [col for col in features_df.columns if 'Target' not in col]
        last_sequence = features_df[feature_columns].iloc[-SEQUENCE_LENGTH:].values
        
        # 將其塑造成模型需要的形狀 (1, 60, n_features)
        last_sequence_reshaped = np.expand_dims(last_sequence, axis=0)
        
        # 3. 進行單日預測
        tomorrow_price = predictor.predict_from_sequences(last_sequence_reshaped)
        print(f"預測明天 {TICKER} 收盤價: {tomorrow_price:.2f}")

        # 4. 進行多日預測
        next_5_days_prices = predictor.predict_multiple_days(last_sequence_reshaped, days=5)
        print(f"預測未來5天 {TICKER} 收盤價:")
        for i, price in enumerate(next_5_days_prices, 1):
            print(f"  第 {i} 天: {price:.2f}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"執行預測時發生錯誤: {e}")
