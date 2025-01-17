import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, MultiHeadAttention, Flatten, Dense, Bidirectional


def genera_grupos_activos(df_train, len_sec=15, train_split=0.2, output='binary', step=2, random=False, future_win = 4):
    X = df_train.values
    y = df_train[output].values
    X_seq = []
    y_seq = []

    for i in range(0, len(X) - len_sec, step):
        if i + len_sec + future_win > len(X):
            break
        X_seq.append(X[i:i+len_sec])
        y_seq.append(y[i+len_sec:i+len_sec+future_win])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    if not random:
        long2 = X_seq.shape[0]
        split_index = int(long2 * train_split)

        X_train = X_seq[:split_index, :, :]
        y_train = y_seq[:split_index, :]

        X_test = X_seq[split_index:, :, :]
        y_test = y_seq[split_index:, :]
    else:
        indices = np.arange(X_seq.shape[0])
        np.random.shuffle(indices)
        
        X_seq = X_seq[indices]
        y_seq = y_seq[indices]
        split_index = int(X_seq.shape[0] * train_split)
        
        X_train = X_seq[:split_index, :, :]
        y_train = y_seq[:split_index, :]
        X_test = X_seq[split_index:, :, :]
        y_test = y_seq[split_index:, :]

    return X_train, y_train, X_test, y_test

def get_model(X_train, y_train):
    inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
    x = Bidirectional(LSTM(50, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = Flatten()(x)

    x = Dense(64, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    outputs = Dense(y_train.shape[1], activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model

def train_model(path, model, X_train, y_train, X_test, y_test, epochs=2000, batch_size=32, lr=0.001):
    
    cp4 = ModelCheckpoint(path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=[RootMeanSquaredError(), 'mae'])
    #history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2500, callbacks=[cp4, early_stopping])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[cp4, early_stopping])
    return model, history