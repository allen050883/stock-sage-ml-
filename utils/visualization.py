# utils/visualization.py
# This file can contain functions to plot predictions, feature importance, etc.
import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title='Stock Price Prediction'):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, color='blue', label='Actual Price')
    plt.plot(y_pred, color='red', linestyle='--', label='Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.show()
