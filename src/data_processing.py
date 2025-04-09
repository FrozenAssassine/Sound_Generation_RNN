import random
import pickle
import os
import numpy as np
from keras.utils import to_categorical
from music21 import converter, note, chord

from typing import cast

def load_training_data(model_data_path):
    with open(model_data_path, "wb") as f:
        return cast(DataProcessing, pickle.load(f))

class DataProcessing:
    def __init__(self) -> None:
        self.unique_notes = []
        self.note_to_int = []
        self.int_to_note = []
        self.all_notes = []
        self.notes_in = []
        self.notes_out = []
        
    def make_lookup_table(self, all_notes):
        self.unique_notes = sorted(set(item for item in all_notes))
        self.note_to_int = dict((note, number) for number, note in enumerate(self.unique_notes))
        self.int_to_note = dict((number, note) for number, note in enumerate(self.unique_notes))


    def make_all_notes(self, midi_files_path, nbr_files):
        progress = 0
        for midi_file in random.choices(os.listdir(midi_files_path), k=nbr_files):  # os.listdir(midi_files_path):
            progress += 1
            print(f"{progress}/{nbr_files}")
            for element in converter.parse(os.path.join(midi_files_path, midi_file)).flat:
                if isinstance(element, note.Note):
                    self.all_notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.all_notes.append('.'.join(str(n) for n in element.normalOrder))


    def make_training_data(self, note_seq_length):
        for i in range(0, len(self.all_notes) - note_seq_length, 1):
            sequence_in = self.all_notes[i:i + note_seq_length]
            sequence_out = self.all_notes[i + note_seq_length]
            self.notes_in.append([self.note_to_int[char] for char in sequence_in])
            self.notes_out.append(self.note_to_int[sequence_out])


    def shape_data(self, note_seq_length):
        n_patterns = len(self.notes_in)
        vocab_size = len(set(self.all_notes))

        X = np.reshape(self.notes_in, (n_patterns, note_seq_length, 1))
        X = X / float(vocab_size)

        y = to_categorical(self.notes_out)

        return X, y, vocab_size


    def get_training_data(self, midi_files_path, note_seq_length, nbr_files):
        self.make_all_notes(midi_files_path, nbr_files)
        self.make_lookup_table(self.all_notes)
        
        
        self.make_training_data(note_seq_length)
        
    def save_data(self, model_data_path):
        with open(model_data_path, "wb") as f:
            pickle.dump(self, f)
     