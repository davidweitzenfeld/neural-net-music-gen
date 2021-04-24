import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl


def create_lstm_rnn_model(input_shape, vocab_size: int):
    model = tfk.Sequential(
        name='lstm_rnn_model',
        layers=[
            tfkl.LSTM(units=128, input_shape=input_shape[1:], return_sequences=True),
            tfkl.Dropout(rate=0.2),
            tfkl.Flatten(),
            tfkl.Dense(units=vocab_size),
            tfkl.Softmax(),
        ]
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy')

    return model
