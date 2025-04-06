from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from data_processing import get_training_data, shape_data
from model import create_model
import matplotlib.pyplot as plt
import config
import tensorflow as tf


pitch_names, all_notes, note_to_int, int_to_note, notes_in, notes_out = get_training_data(config.MIDIPATH, config.NOTE_SEQUENCE_LENGTH)
X, y, vocab_size = shape_data(notes_in, notes_out, all_notes, config.NOTE_SEQUENCE_LENGTH)

model = create_model(X, vocab_size)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.96)

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(learning_rate=lr_schedule),  metrics=['accuracy'])


history = model.fit(X, y, epochs=config.EPOCHS, batch_size=config.BATCHES)
model.save(config.MODEL_PATH)


plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
