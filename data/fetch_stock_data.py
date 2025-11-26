# data/fetch_stock_data.py
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    """
    下載台股歷史數據
    ticker: 股票代碼，需加上 .TW 後綴，如 '2330.TW'
    """
    stock = yf.Ticker(ticker)
    
    # 獲取歷史價格
    df = stock.history(start=start_date, end=end_date)
    
    # 獲取基本面數據
    info = stock.info
    
    return df, info

def fetch_multiple_stocks(tickers, start_date, end_date):
    """批量下載多支股票"""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

# 範例使用
# 以台積電, 聯發科, 鴻海, 台達電, 富邦金 為例
tickers = ['2330.TW', '2454.TW', '2317.TW', '2308.TW', '2881.TW']
# 注意：yf.download對於多個台股代碼的處理可能不穩定，建議單獨下載後合併
data = fetch_multiple_stocks(tickers, '2020-01-01', '2024-11-20')
print("已下載台股數據:")
print(data.head())
