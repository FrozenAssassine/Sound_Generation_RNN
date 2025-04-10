from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy

from data_processing import DataProcessing, load_training_data
from model import create_model
import matplotlib.pyplot as plt
import config
import tensorflow as tf

# get the data:

# data_processing = DataProcessing()
# data_processing.get_training_data(config.MIDIPATH, config.NOTE_SEQUENCE_LENGTH, config.TRAINING_FILE_COUNT)
# data_processing.save_data(config.MODEL_DATA_PATH)

data_processing = load_training_data(config.MODEL_DATA_PATH)
X, y, vocab_size = data_processing.shape_data(config.NOTE_SEQUENCE_LENGTH)

model = create_model(X, vocab_size)

model.load_weights(config.MODEL_PATH)

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
