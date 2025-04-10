import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, InputLayer, BatchNormalization


def predict_notes(model, notes_in, vocab_size, int_to_note, song_length):
    start = np.random.randint(0, len(notes_in)-1)
    pattern = notes_in[start]

    prediction_output = []
    for _ in range(song_length):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)

        prediction_index = np.argmax(model.predict(prediction_input, verbose=1))
        result = int_to_note[prediction_index]

        prediction_output.append(result)
        pattern.append(prediction_index)
        pattern = pattern[1:len(pattern)]

    return prediction_output, start, pattern


def create_model(input_data, vocab_size):
    model = Sequential()
    model.add(InputLayer(shape=(input_data.shape[1], input_data.shape[2])))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))

    return model
