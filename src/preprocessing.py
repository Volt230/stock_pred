# src/preprocessing.py

import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler


# ------------------------------------------------------
# SCALE TRAINING DATA (X sequence + y target)
# ------------------------------------------------------
def scale_data(X, y):
    """
    Scales X (3D sequences) and y (prices).
    Returns: X_scaled, y_scaled, scaler_X, scaler_y
    """

    # Flatten X for scaler
    N, seq_len, feat = X.shape
    X_flat = X.reshape(N, seq_len * feat)

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X_flat)
    y_scaled = scaler_y.fit_transform(y)

    # Reshape back into LSTM input form
    X_scaled = X_scaled.reshape(N, seq_len, feat)

    return X_scaled, y_scaled, scaler_X, scaler_y


# ------------------------------------------------------
# SAVE SCALER
# ------------------------------------------------------
def save_scaler(scaler, path):
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


# ------------------------------------------------------
# LOAD SCALER
# ------------------------------------------------------
def load_scaler(path):
    with open(path, "rb") as f:
        return pickle.load(f)
