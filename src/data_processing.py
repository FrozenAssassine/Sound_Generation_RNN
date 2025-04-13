import random
import pickle
import os
import numpy as np
from keras.utils import to_categorical
from music21 import converter, note, chord

from typing import cast


def load_training_data(model_data_path):
    with open(model_data_path, "rb") as f:
        return cast(DataProcessing, pickle.load(f))


class DataProcessing:
    def __init__(self) -> None:
        self.unique_notes = []
        self.note_to_int = []
        self.int_to_note = []
        self.all_notes_by_style = {}
        self.notes_in = []
        self.notes_out = []
        self.styles = []
        self.style_vectors = []
        self.style_to_vector = {}

    def make_lookup_table(self):
        all_notes_flat = [note for notes in self.all_notes_by_style.values() for note in notes]
        self.unique_notes = sorted(set(all_notes_flat))

        self.note_to_int = dict((note, number) for number, note in enumerate(self.unique_notes))
        self.int_to_note = dict((number, note) for number, note in enumerate(self.unique_notes))

        self.style_to_vector = {
            style: [1 if i == idx else 0 for i in range(len(self.styles))]
            for idx, style in enumerate(self.styles)
        }

    def get_styles_from_folders(self, midi_files_path):
        for style_folder in os.listdir(midi_files_path):
            self.styles.append(style_folder)

    def make_all_notes(self, style_name, midi_files_path, nbr_files):
        self.all_notes_by_style[style_name] = []

        progress = 0
        for midi_file in random.choices(os.listdir(os.path.join(midi_files_path, style_name)), k=nbr_files):  # os.listdir(midi_files_path):
            progress += 1
            print(f"{progress}/{nbr_files}")
            try:
                for element in converter.parse(os.path.join(midi_files_path, style_name, midi_file)).flat:
                    if isinstance(element, note.Note):
                        self.all_notes_by_style[style_name].append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        self.all_notes_by_style[style_name].append('.'.join(str(n) for n in element.normalOrder))
            except:
                print("Could not convert file")

    def make_training_data(self, note_seq_length):
        for style, notes in self.all_notes_by_style.items():
            for i in range(0, len(notes) - note_seq_length, 1):
                sequence_in = notes[i:i + note_seq_length]
                sequence_out = notes[i + note_seq_length]

                self.notes_in.append([self.note_to_int[n] for n in sequence_in])
                self.notes_out.append(self.note_to_int[sequence_out])
                self.style_vectors.append(self.style_to_vector[style])

    def shape_data(self, note_seq_length):
        n_patterns = len(self.notes_in)
        vocab_size = len(self.unique_notes)

        X = np.reshape(self.notes_in, (n_patterns, note_seq_length, 1))
        X = X / float(vocab_size)

        y = to_categorical(self.notes_out)
        style_input = np.array(self.style_vectors)

        indices = np.arange(n_patterns)
        np.random.shuffle(indices)

        X = X[indices]
        y = y[indices]
        style_input = style_input[indices]

        return X, style_input, y, vocab_size

    def get_training_data(self, midi_files_path, note_seq_length, nbr_files):
        self.get_styles_from_folders(midi_files_path)

        for style_folder in self.styles:
            self.make_all_notes(style_folder, midi_files_path, nbr_files)

        self.make_lookup_table()

        self.make_training_data(note_seq_length)

    def save_data(self, model_data_path):
        with open(model_data_path, "wb") as f:
            pickle.dump(self, f)
            print("data saved")
