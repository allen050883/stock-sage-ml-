# utils/backtest.py
# This file can contain a simple backtesting engine to evaluate
# a trading strategy based on the model's predictions.
import pandas as pd

def simple_backtest(prices, signals):
    """
    A simple backtesting engine.
    :param prices: pd.Series of actual prices.
    :param signals: pd.Series of trading signals (1 for buy, -1 for sell, 0 for hold).
    """
    initial_capital = 100000.0
    positions = pd.DataFrame(index=signals.index).fillna(0.0)
    portfolio = pd.DataFrame(index=signals.index).fillna(0.0)

    # Buy/Sell based on signal
    positions['stock'] = signals.cumsum()
    
    # Calculate portfolio value
    portfolio['positions'] = positions.multiply(prices, axis=0)
    portfolio['cash'] = initial_capital - (signals.multiply(prices)).cumsum()
    portfolio['total'] = portfolio['positions'] + portfolio['cash']
    
    returns = portfolio['total'].pct_change()
    
    print(f"Final portfolio value: ${portfolio['total'][-1]:.2f}")
    print(f"Total return: {((portfolio['total'][-1] / initial_capital) - 1) * 100:.2f}%")
    
    return portfolio
