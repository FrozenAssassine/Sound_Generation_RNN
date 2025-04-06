import os
import numpy as np
from keras.utils import to_categorical
from music21 import converter, note, chord
import random


def make_lookup_table(all_notes):
    pitch_names = sorted(set(item for item in all_notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    return pitch_names, note_to_int, int_to_note


def make_all_notes(midi_files_path):
    all_notes = []
    progress = 0
    for midi_file in random.choices(os.listdir(midi_files_path), k=100):  # os.listdir(midi_files_path):
        progress += 1
        print(f"{progress}/100")
        for element in converter.parse(os.path.join(midi_files_path, midi_file)).flat:
            if isinstance(element, note.Note):
                all_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                all_notes.append('.'.join(str(n) for n in element.normalOrder))

    return all_notes


def make_training_data(all_notes, note_to_int, note_seq_length):
    notes_in = []
    notes_out = []

    for i in range(0, len(all_notes) - note_seq_length, 1):
        sequence_in = all_notes[i:i + note_seq_length]
        sequence_out = all_notes[i + note_seq_length]
        notes_in.append([note_to_int[char] for char in sequence_in])
        notes_out.append(note_to_int[sequence_out])

    return notes_in, notes_out


def shape_data(notes_in, notes_out, all_notes, note_seq_length):
    n_patterns = len(notes_in)
    vocab_size = len(set(all_notes))

    X = np.reshape(notes_in, (n_patterns, note_seq_length, 1))
    X = X / float(vocab_size)

    y = to_categorical(notes_out)

    return X, y, vocab_size


def get_training_data(midi_files_path, note_seq_length):
    all_notes = make_all_notes(midi_files_path)
    pitch_names, note_to_int, int_to_note = make_lookup_table(all_notes)

    notes_in, notes_out = make_training_data(all_notes, note_to_int, note_seq_length)

    return pitch_names, all_notes, note_to_int, int_to_note, notes_in, notes_out
