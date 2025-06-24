from keras.models import Sequential
from keras.layers import LSTM, Dense, Reshape, Input
from keras.callbacks import EarlyStopping

class NextImagePredictor(Sequential):
    def __init__(self, input_shape):
        super().__init__()
        self.add(Input(shape=input_shape))
        self.add(LSTM(128, activation='relu', return_sequences=True))
        self.add(LSTM(64, activation='relu'))
        self.add(Dense(28*28, activation='linear'))
        self.add(Reshape((28, 28)))

    def train_model(self, X_train, y_train, optimizer='adam', epochs=10):
        self.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
        early_stop = EarlyStopping(monitor='loss', patience=3)
        history = self.fit(X_train, y_train, epochs=epochs, callbacks=[early_stop])
        return history