# models/xgboost_model.py
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

def train_xgboost(X_train, y_train, X_test, y_test):
    """
    訓練XGBoost模型
    """
    # XGBoost needs 2D data, not the 3D sequences for LSTM
    if X_train.ndim > 2:
        nsamples, nx, ny = X_train.shape
        X_train_2d = X_train.reshape((nsamples,nx*ny))
    else:
        X_train_2d = X_train

    if X_test.ndim > 2:
        nsamples, nx, ny = X_test.shape
        X_test_2d = X_test.reshape((nsamples,nx*ny))
    else:
        X_test_2d = X_test

    dtrain = xgb.DMatrix(X_train_2d, label=y_train)
    dtest = xgb.DMatrix(X_test_2d, label=y_test)
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.01,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'rmse'
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    return model
