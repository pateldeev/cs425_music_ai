import glob
import os
import pickle

from music21 import converter, instrument, note, chord
import numpy as np
import tensorflow as tf

USE_GPU = True
USE_CACHE = True
TRAIN_NETWORK = True
TMP_DIR = 'tmp'

# Input music params.
CACHE_FILE = "notes"
INPUT_MIDI_FILES = glob.glob("input_songs/*.mid")

# Model params.
LSTM_SEQ_LENGTH = 100
NUM_EPOCHS = 10
BATCH_SIZE = 128
TRAINING_WEIGHTS_FN = "weights_training_E{epoch:04d}_L{loss:.4f}.hdf5"
FINAL_WEIGHTS_FILE = "weights_trained.hdf5"

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_notes_from_midi(fns):
    """ Get all the notes and chords from the midi files. """
    notes = []

    for fn in fns:
        midi = converter.parse(fn)

        print("Parsing file: ", fn)

        try:  # File has instrument parts.
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # File has notes in a flat structure.
            notes_to_parse = midi.flat.notes

        for ele in notes_to_parse:
            if isinstance(ele, note.Note):
                notes.append(str(ele.pitch))
            elif isinstance(ele, chord.Chord):
                notes.append('.'.join(str(n) for n in ele.normalOrder))
            else:
                # TODO(deev): Handle Instrument Types.
                continue

    return notes


def generate_data_sequences(notes, pitches):
    """ Generates the data sequences used by the Neural Network.
    Returns (network inputs raw (unscaled), network inputs, network outputs or labels) """
    # Map between notes and integers.
    note_to_int = {
        note: num for num, note in enumerate(pitches)
    }

    # Create raw inputs and outputs based on sequences.
    net_in_raw, net_out_raw = [], []
    for i in range(len(notes) - LSTM_SEQ_LENGTH):
        seq_in = notes[i: i + LSTM_SEQ_LENGTH]
        seq_out = notes[i + LSTM_SEQ_LENGTH]

        net_in_raw.append(np.array([note_to_int[note] for note in seq_in], dtype=np.float))
        net_out_raw.append(note_to_int[seq_out])

    # Reshape and normalize the input into a format compatible with LSTM layers.
    net_in = np.reshape(net_in_raw, (len(net_in_raw), LSTM_SEQ_LENGTH, 1))
    net_in /= len(pitches)

    # Categorize outputs.
    net_out = tf.keras.utils.to_categorical(net_out_raw)

    return net_in, net_out


def create_model(net_in, pitches):
    """ Create the tensorflow LSTM based model. """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        512,
        input_shape=(net_in.shape[1], net_in.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(tf.keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(len(pitches)))
    model.add(tf.keras.layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train(model, x, y, batch_size, epochs):
    """ train the model. """

    # Delete any older weights files.
    for f in glob.glob(os.path.join(TMP_DIR, TRAINING_WEIGHTS_FN[:TRAINING_WEIGHTS_FN.find('{')])):
        os.remove(f)

    # Define checkpoint to save weights during training.
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(TMP_DIR, TRAINING_WEIGHTS_FN),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
    )

    # Train model.
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, callbacks=[ckpt])


if __name__ == '__main__':
    print("Start")

    # Create directory for notes cache and weights.
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    # Get notes.
    if not USE_CACHE:
        notes = get_notes_from_midi(fns=INPUT_MIDI_FILES)
        with open(os.path.join(TMP_DIR, CACHE_FILE), "wb") as f:
            pickle.dump(notes, f)
    else:
        with open(os.path.join(TMP_DIR, CACHE_FILE), "rb") as f:
            notes = pickle.load(f)
    print(notes)

    # Orderd list of unique pitches.
    pitches = sorted(set(notes))
    print(pitches)

    # Get network inputs and outputs (labels).
    net_in, net_out = generate_data_sequences(notes=notes, pitches=pitches)
    print("Network inputs: ", net_in)
    print("Network ouputs: ", net_out)

    # Create model.
    model = create_model(net_in=net_in, pitches=pitches)
    print(model.summary())

    # Train model or load weights from file.
    if TRAIN_NETWORK:
        train(model=model, x=net_in, y=net_out, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
        model.save_weights(os.path.join(TMP_DIR, FINAL_WEIGHTS_FILE))
    else:
        model.load_weights(os.path.join(TMP_DIR, FINAL_WEIGHTS_FILE))
    print("Got trained model!")

    print("End")
