from music21 import instrument, note, chord, stream

from data_processing import load_training_data
from model import create_model, predict_notes
import config

# get the data:
data_processing = load_training_data(config.MODEL_DATA_PATH)
X_notes, X_style, y, vocab_size = data_processing.shape_data(config.NOTE_SEQUENCE_LENGTH)

model = create_model(X_notes.shape[1:], vocab_size, X_style.shape[1])
model.summary()

model.load_weights(config.MODEL_PATH)

print(data_processing.styles)

prediction_output = predict_notes(model, data_processing.notes_in, vocab_size, data_processing.int_to_note, config.SONG_LENGTH, [1, 1, 1])


def make_note(current_note):
    try:
        return note.Note(int(current_note))
    except ValueError:
        return note.Note(current_note)


offset = 0
output_notes = []
for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')

        notes = []
        for current_note in notes_in_chord:
            next_note = make_note(current_note)
            next_note .storedInstrument = instrument.Piano()
            notes.append(next_note)

        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        next_note = make_note(pattern)
        next_note.offset = offset
        next_note.storedInstrument = instrument.Piano()
        output_notes.append(next_note)
    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp=config.OUTPUT_PATH)
