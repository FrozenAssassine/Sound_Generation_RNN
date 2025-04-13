import os
import numpy as np
from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, Concatenate


def predict_notes(model, notes_in, vocab_size, int_to_note, song_length, style_vector):
    start = np.random.randint(0, len(notes_in)-1)
    pattern = notes_in[start]  # list of ints

    prediction_output = []

    for _ in range(song_length):
        # Reshape and normalize notes input
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(vocab_size)

        # Expand style vector to batch size 1
        style_input = np.reshape(style_vector, (1, len(style_vector)))

        # Predict with both inputs
        prediction_probs = model.predict([prediction_input, style_input], verbose=1)
        prediction_index = np.argmax(prediction_probs)
        result = int_to_note[prediction_index]

        prediction_output.append(result)
        pattern.append(prediction_index)
        pattern = pattern[1:]  # shift the window

    return prediction_output


def create_model(input_shape, vocab_size, style_vector_size):
    note_input = Input(shape=(input_shape))
    style_input = Input(shape=(style_vector_size,))

    style_repeated = RepeatVector(input_shape[0])(style_input)

    # Merge style and note sequence
    merged_input = Concatenate(axis=-1)([note_input, style_repeated])

    # LSTM layers
    x = LSTM(512, return_sequences=True)(merged_input)
    x = Dropout(0.2)(x)
    x = LSTM(512)(x)
    x = Dense(256)(x)
    x = Dropout(0.2)(x)
    output = Dense(vocab_size, activation='softmax')(x)

    model = Model(inputs=[note_input, style_input], outputs=output)

    return model
