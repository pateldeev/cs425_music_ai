import glob
import os
import pickle
import sys

from music21 import chord, converter, instrument, note, stream
import numpy as np
import tensorflow as tf

# TODO - Convert these to command line flags.
USE_GPU = True
USE_CACHE = True
TRAIN_NETWORK = False
TMP_DIR = 'tmp'

# Input music params.
CACHE_FILE = "notes"
INPUT_MIDI_FILES = glob.glob("input_songs/*.mid")

# Model params.
LSTM_SEQ_LENGTH = 100
NUM_EPOCHS = 200
BATCH_SIZE = 128
TRAINING_WEIGHTS_FN = "weights_training_E{epoch:04d}_L{loss:.4f}.hdf5"
FINAL_WEIGHTS_FILE = "weights_trained.hdf5"
OUTPUT_NOTES = 500
OUTPUT_FILE = "model_output.mid"

userLoop = False

# gives user option to change model parameters
tempInput = input("Do you want to change any model parameters? y/n\n").lower()
if tempInput == 'y':
    userLoop = True

# loops until all desired parameters have been changed
while userLoop:
    # lists all changeable parameters
    print("Enter the number of the parameter you want to change")
    print("1. USE_GPU: " + str(USE_GPU))
    print("2. USE_CACHE: " + str(USE_CACHE))
    print("3. TRAIN_NETWORK: " + str(TRAIN_NETWORK))
    print("4. TMP_DIR: " + str(TMP_DIR))
    print("5. CACHE_FILE: " + str(CACHE_FILE))
    print("6. INPUT_MIDI_FILES: " + str(INPUT_MIDI_FILES))
    print("7. LSTM_SEQ_LENGTH: " + str(LSTM_SEQ_LENGTH))
    print("8. NUM_EPOCHS: " + str(NUM_EPOCHS))
    print("9. BATCH_SIZE: " + str(BATCH_SIZE))
    print("10. TRAINING_WEIGHTS_FN: " + str(TRAINING_WEIGHTS_FN))
    print("11. FINAL_WEIGHTS_FILE: " + str(FINAL_WEIGHTS_FILE))
    print("12. OUTPUT_NOTES: " + str(OUTPUT_NOTES))
    print("13. OUTPUT_FILE: " + str(OUTPUT_FILE))
    print("14. Exit")
    paramChange = input()

    # compares user input to possible options, then allows the to change the requested flag
    if paramChange == '1':
        USE_GPU = input("Enter the new parameter value: ")
    elif paramChange == '2':
        USE_CACHE = input("Enter the new parameter value: ")
    elif paramChange == '3':
        TRAIN_NETWORK = input("Enter the new parameter value: ")
    elif paramChange == '4':
        TMP_DIR = input("Enter the new parameter value: ")
    elif paramChange == '5':
        CACHE_FILE = input("Enter the new parameter value: ")
    elif paramChange == '6':
        INPUT_MIDI_FILES += [input("Enter the location of the file you want to add: ")]
    elif paramChange == '7':
        LSTM_SEQ_LENGTH = int(input("Enter the new parameter value: "))
    elif paramChange == '8':
        NUM_EPOCHS = int(input("Enter the new parameter value: "))
    elif paramChange == '9':
        BATCH_SIZE = int(input("Enter the new parameter value: "))
    elif paramChange == '10':
        TRAINING_WEIGHTS_FN = input("Enter the new parameter value: ")
    elif paramChange == '11':
        FINAL_WEIGHTS_FILE = input("Enter the new parameter value: ")
    elif paramChange == '12':
        OUTPUT_NOTES = int(input("Enter the new parameter value: "))
    elif paramChange == '13':
        OUTPUT_FILE = input("Enter the new parameter value: ")
    elif paramChange == '14':
        userLoop = False
    else:
        print("\nInvalid Input, please select another option")

if not USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
TF_DEVICES = tf.config.list_physical_devices()


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

    return net_in_raw, net_in, net_out


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
    """ Train the model. """

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


def predict(model, network_in_raw, pitches):
    """ Generate notes from the neural network. """
    # Pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_in_raw)-1)

    # Map between notes and integers.
    int_to_note = {
        num: note for num, note in enumerate(pitches)
    }

    pattern = list(network_in_raw[start])
    model_out = []

    # Generate notes.
    for ni in range(OUTPUT_NOTES):
        print("Generating note: ", ni)
        net_in = np.reshape(pattern, (1, len(pattern), 1))
        net_in = net_in / float(len(pitches))

        y_hat = model.predict(net_in, verbose=0)

        i = np.argmax(y_hat)
        model_out.append(int_to_note[i])

        pattern.append(i)
        pattern.pop(0)

    return model_out


def create_midi(prediction_data):
    """ Convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_data:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # TODO(deev): Handle Instrument Types.

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=OUTPUT_FILE)
    print("Saved prediciton to: ", OUTPUT_FILE)


if __name__ == '__main__':
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nStart")

    # Print system and device information.
    print("Python version: ", sys.version)
    print("Version info:", sys.version_info)
    print("Tensorflow version:", tf.__version__)
    print("Tensorflow Devices:", TF_DEVICES)

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
    # print(notes)

    # Orderd list of unique pitches.
    pitches = sorted(set(notes))
    # print(pitches)

    # Get network inputs and outputs (labels).
    net_in_raw, net_in, net_out = generate_data_sequences(notes=notes, pitches=pitches)
    # print("Network inputs: ", net_in)
    # print("Network ouputs: ", net_out)

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

    # Generation prediction of model.
    model_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches)
    print("Predicted Notes:", model_prediction)

    # Save prediction as midi file.
    create_midi(prediction_data=model_prediction)

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nEnd")
