# 🎵 Midi Music Generation Neural Network using RNN (LSTM Layer)

## 🧠 How Does It Work?

The MIDI files are loaded and parsed using the `music21` Python library.  
All the notes are tokenized and stored. Every note is stored in a mapping table: `notes_to_int` and `int_to_notes`.

The training data is created from the notes. The neural network is trained on 100 notes as input and the next following note as the output.  
The LSTM layer captures the important data and sequences during training.

The model works by predicting the next note after a sequence.  
This means, if you enter a sequence, the next note gets predicted and added to the input sequence.  
Then, the new input sequence is fed through the neural network again, and the next fitting note is returned.

This process continues for as long as specified. For testing, I specified 200 steps.

After that, you have a file with all the generated notes, which you can open in any DAW or similar software.  
I used **FL Studio** to test the output.

## 🧰 Technologies Used

- Python
- TensorFlow / Keras (RNN with LSTM Layer)
- music21

## 🎶 Example Audio from my Model:

[](https://github.com/user-attachments/assets/85047b89-8b28-4c3e-bb40-06efef766bc7)

