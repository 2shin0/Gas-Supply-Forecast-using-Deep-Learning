import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping

def train_lstm_model(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    window_size = 30
    X, y = create_sequences(scaled_data, window_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        LSTM(50, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])

    return model, history

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :-1])
        y.append(data[i + window_size, -1])
    return np.array(X), np.array(y)