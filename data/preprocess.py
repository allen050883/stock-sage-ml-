# data/preprocess.py
import pandas as pd
import numpy as np
import ta  # Technical Analysis library

def create_features(df):
    """創建技術指標特徵"""
    
    # 基本價格特徵
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 移動平均線
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # 技術指標
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['BB_upper'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['BB_lower'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    
    # 成交量特徵
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    # 波動率
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    
    # 價格區間
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # 時間特徵
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    
    # 目標變數（預測未來N天的漲跌）
    df['Target_1d'] = df['Close'].shift(-1)  # 明天收盤價
    df['Target_5d'] = df['Close'].shift(-5)  # 5天後收盤價
    df['Target_Direction'] = (df['Target_1d'] > df['Close']).astype(int)
    
    return df.dropna()

def add_sentiment_features(df, news_df):
    """
    加入情緒分析特徵（可選）
    news_df: 包含日期和情緒分數的DataFrame
    """
    df = df.merge(news_df, left_index=True, right_on='date', how='left')
    return df

def prepare_sequences(df, sequence_length=60):
    """
    為LSTM模型準備時間序列數據
    sequence_length: 使用過去N天的數據來預測
    """
    feature_columns = [col for col in df.columns 
                      if col not in ['Target_1d', 'Target_5d', 'Target_Direction']]
    
    X, y = [], []
    
    for i in range(sequence_length, len(df)):
        X.append(df[feature_columns].iloc[i-sequence_length:i].values)
        y.append(df['Target_1d'].iloc[i])
    
    return np.array(X), np.array(y)

def train_test_split_time_series(X, y, test_size=0.2):
    """時間序列專用的分割（不能隨機打亂）"""
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test
