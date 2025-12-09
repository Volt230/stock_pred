# src/models.py
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model

def build_lstm_multihead(seq_len, n_features):
    inp = Input(shape=(seq_len, n_features))
    x = LSTM(128, return_sequences=False)(inp)
    x = Dropout(0.2)(x)
    # price regression head
    price = Dense(64, activation="relu")(x)
    price_out = Dense(1, name="price_output")(price)
    # trend classification head (prob)
    trend = Dense(32, activation="relu")(x)
    trend_out = Dense(1, activation="sigmoid", name="trend_output")(trend)
    model = Model(inputs=inp, outputs=[price_out, trend_out])
    model.compile(optimizer="adam",
                  loss={"price_output":"mse","trend_output":"binary_crossentropy"},
                  loss_weights={"price_output":1.0,"trend_output":0.2})
    return model
