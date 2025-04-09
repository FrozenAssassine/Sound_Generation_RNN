#Model
MODEL_NAME = "model1"
SONG_LENGTH = 200
EPOCHS = 45
BATCHES = 64
NOTE_SEQUENCE_LENGTH = 100

#Data
MIDIPATH = '/mnt/w/Documents/NN DATASETS/MidiSounds/2' # '/mnt/f/NN DATASETS/MidiSounds/midi_dataset/2'
TRAINING_FILE_COUNT = 5

#Out
OUTPUT_PATH = "../output.mid"
MODEL_PATH = f"../{MODEL_NAME}.keras"
MODEL_DATA_PATH = f"../{MODEL_NAME}.pkl"