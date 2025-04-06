from music21 import converter, instrument, note, chord, stream

from data_processing import get_training_data, shape_data
from model import create_model, predict_notes
import config

pitch_names, all_notes, note_to_int, int_to_note, notes_in, notes_out = get_training_data(config.MIDIPATH, config.NOTE_SEQUENCE_LENGTH)
X, y, vocab_size = shape_data(notes_in, notes_out, all_notes, config.NOTE_SEQUENCE_LENGTH)

model = create_model(X, vocab_size)
model.summary()

model.load_weights(config.MODEL_PATH)

prediction_output, start, pattern = predict_notes(model, notes_in, vocab_size, int_to_note, config.SONG_LENGTH)


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
