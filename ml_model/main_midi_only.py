# Training script that only works with midi files of piano music.

import argparse
import glob
import os
import pickle
import shutil
import sys
import wave
from collections import defaultdict

import musdb
import numpy as np
import tensorflow as tf
from midi2audio import FluidSynth
from music21 import chord, converter, instrument, note, stream
from pydub import AudioSegment

parser = argparse.ArgumentParser(description="Train ML music model")

# Setup Params.
parser.add_argument("--no_gpu", action="store_true")
parser.add_argument("--use_cache", action="store_true")
parser.add_argument("--no_train", action="store_true")

# Input Params.
parser.add_argument("--tmp_dir", type=str, default="/home/dp/Desktop/cs425_music_ai/ml_model/tmp")
parser.add_argument("--cache_file", type=str, default="notes_cache")
parser.add_argument("--input_files_glob", type=str, default="/home/dp/Desktop/cs425_music_ai/ml_model/input_songs/*.mid")
parser.add_argument("--input_files_override", type=str, nargs="+")

# Training Params
parser.add_argument("--lstm_seq_len", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)

# Output Params.
parser.add_argument("--final_weights_file", type=str, default="weights_trained.hdf5")
parser.add_argument("--output_notes", type=int, default=50)
parser.add_argument("--output_file", type=str, default="/home/dp/Desktop/cs425_music_ai/ml_model/model_output.mid")

# Parse command line arguments.
args = parser.parse_args()

# Configure GPU.
if args.no_gpu:
    # Don't use GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # Prevent GPU memory overflow during training.
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], enable=True)

# Constants.
TRAINING_WEIGHTS_FN = "weights_training_E{epoch:04d}_L{loss:.4f}.hdf5"
TF_DEVICES = tf.config.list_physical_devices()


def get_notes_from_midi(fns):
    """
    Get all the notes and chords from the midi files.
    Param: List of files to parse
    Return: 2-D list of notes. One list of notes per input song
    """
    notes = []

    for fn in sorted(fns):
        notes.append([])
        if not fn.endswith((".mid", ".midi")):
            # Only support midi files.
            continue

        # Copy file to temporary directory.
        fn_tmp = os.path.join(args.tmp_dir, fn[fn.rfind("/") + 1 :]) + ".mid"
        shutil.copyfile(fn, fn_tmp)
        midi = converter.parse(fn_tmp)

        # Prase file.
        print("Parsing input file: ", fn)
        sys.stdout.flush()
        try:
            # File has instrument parts.
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            # File has notes in a flat structure.
            notes_to_parse = midi.flat.notes

        # Make note. This version only supports piano chords.
        for ele in notes_to_parse:
            if isinstance(ele, note.Note):
                notes[-1].append(str(ele.pitch))
            elif isinstance(ele, chord.Chord):
                notes[-1].append(".".join(str(n) for n in ele.normalOrder))
            else:
                # TODO(deev): Handle more instrument and vocal types.
                continue

    return notes


def generate_data_sequences(notes, pitches):
    """
    Generates the data sequences used by the Neural Network.
    Param: (2D list of notes, set of pitches)
    Return: (network inputs raw (unscaled), network inputs (scaled), network outputs or labels)
    """
    # Map between notes and integers.
    note_to_int = {note: num for num, note in enumerate(pitches)}

    # Create raw inputs and outputs based on sequences.
    net_in_raw, net_out_raw = [], []
    for i in range(len(notes) - args.lstm_seq_len):
        seq_in = notes[i : i + args.lstm_seq_len]
        seq_out = notes[i + args.lstm_seq_len]

        net_in_raw.append(np.array([note_to_int[note] for note in seq_in], dtype=np.float))
        net_out_raw.append(note_to_int[seq_out])

    # Reshape and normalize the input into a format compatible with LSTM layers.
    net_in = np.reshape(net_in_raw, (len(net_in_raw), args.lstm_seq_len, 1))
    net_in /= len(pitches)

    # Categorize outputs.
    net_out = tf.keras.utils.to_categorical(net_out_raw)

    return net_in_raw, net_in, net_out


def create_model(net_in, pitches):
    """
    Create the tensorflow LSTM based model.
    Param: (network inputs, set of pitches)
    Output: Tensorflow keras LSTM model
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.LSTM(
            512, input_shape=(net_in.shape[1], net_in.shape[2]), recurrent_dropout=0.3, return_sequences=True
        )
    )
    model.add(tf.keras.layers.LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(tf.keras.layers.LSTM(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(len(pitches)))
    model.add(tf.keras.layers.Activation("softmax"))

    # Use |categorical_crossentropy| loss functionand |rmsprop| optimizer.
    # TODO(deev): Experiment with more optimizers and training params (like learning rate) to see if we do better.
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

    return model


def train(model, x, y, batch_size, epochs):
    """
    Train the model.
    Param: (the tensorflow model, list of examples, list of labels for inputs, batch size, number of epochs to train for)
    Return: None. The input model will now be trained.
    """

    # Delete any older weights files.
    for f in glob.glob(os.path.join(args.tmp_dir, TRAINING_WEIGHTS_FN[: TRAINING_WEIGHTS_FN.find("{")])):
        os.remove(f)

    # Define checkpoint to save weights during training.
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.tmp_dir, TRAINING_WEIGHTS_FN),
        save_freq=len(x),
        monitor="loss",
        verbose=0,
        save_best_only=True,
        mode="min",
    )

    # Train model.
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, callbacks=[ckpt])


def predict(model, network_in_raw, pitches, num_notes):
    """
    Generates predictions from the neural network. Creates a new song.
    Param: (trained model, list of unscaled network inputs used for training, set of pitches, number of notes to generate)
    Return: list of new notes.
    """
    # Pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_in_raw) - 1)

    # Map between notes and integers.
    int_to_note = {num: note for num, note in enumerate(pitches)}

    pattern = list(network_in_raw[start])
    model_out = []

    # Generate notes.
    for ni in range(num_notes):
        print("Generating note: ", ni)
        sys.stdout.flush()

        net_in = np.reshape(pattern, (1, len(pattern), 1))
        net_in = net_in / float(len(pitches))

        y_hat = model.predict(net_in, verbose=0)

        i = np.argmax(y_hat)
        model_out.append(int_to_note[i])

        pattern.append(i)
        pattern.pop(0)

    return model_out


def create_output_notes(prediction_data):
    """
    Creates output notes from the raw network output created by a call to |Predict|.
    Param: notes generated by model
    Return: Notes compatible with midi files.
    """
    offset = 0
    output_notes = []

    # Create note and chord objects based on the values generated by the model.
    for pattern in prediction_data:
        if ("." in pattern) or pattern.isdigit():
            # Pattern is a chord.
            notes_in_chord = pattern.split(".")
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack.
        offset += 0.5

    return output_notes


#
def check_uniqueness(song_bank, test_song):
    """
    Loops through all songs and generates uniqueness score for each. Returns worst performance.
    Param: (2D song bank with notes for each song. test song for similarity analysis)
    Return: The similarity to the song that is closest to the test song.
    """
    # Used to disregards songs that are too short to impactfully compare.
    check_notes = 0
    check_counts = defaultdict(int)

    # Find frequency of all notes in generated song.
    for song_note in test_song:
        check_counts[song_note] += 1
        check_notes += 1

    worst_score = 0.0
    # Go through each song that was used for training and compare each note frequency to the generated song.
    # Keeps the score of the song that is most similar to the generated song by this metric, which will be
    # used for filtering later.
    for train_song in song_bank:
        # Tracks number and type of notes for train song.
        test_counts = {}
        test_notes = 0

        # Finds frequency of all notes in generated song.
        for song_note in train_song:
            if song_note not in test_counts:
                test_counts[song_note] = 0

            test_counts[song_note] = test_counts[song_note] + 1
            test_notes += 1

        sim_score = 1.0
        # Compare frequencies of notes.
        if len(train_song) > 40:
            for note_type in test_counts:
                if note_type in check_counts:
                    sim_score -= abs(test_counts[note_type] / test_notes - check_counts[note_type] / check_notes)
                else:
                    sim_score -= test_counts[note_type] / test_notes
        else:
            sim_score = 0.0
        worst_score = max(worst_score, sim_score)

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("Similarity Score: ", worst_score)
    sys.stdout.flush()
    return worst_score


if __name__ == "__main__":
    print("Command: ", " ".join(["python3", "/home/dp/Desktop/cs425_music_ai/ml_model/main_old.py"] + sys.argv[1:]))
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nStart")

    # Print system and device information.
    print("Python version: ", sys.version)
    print("Version info:", sys.version_info)
    print("Tensorflow version:", tf.__version__)
    print("Tensorflow Devices:", TF_DEVICES)
    sys.stdout.flush()

    # Create directory for notes cache and weights.
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)

    # Get notes.
    print("||||||||||||||||||||||||||||||||||||||| Getting Notes")
    if args.use_cache:
        with open(os.path.join(args.tmp_dir, args.cache_file), "rb") as f:
            notes_2d = pickle.load(f)
    else:
        input_files = args.input_files_override if args.input_files_override else glob.glob(args.input_files_glob)
        assert input_files, "No input files!"
        notes_2d = get_notes_from_midi(fns=input_files)

        # Save to cache. Useful for debugging when running script multiple times on same inputs.
        with open(os.path.join(args.tmp_dir, args.cache_file), "wb") as f:
            pickle.dump(notes_2d, f)
    print("||||||||||||||||||||||||||||||||||||||| Done Getting Notes")
    # print(notes_2d)
    notes = []
    for notes_per_song in notes_2d:
        notes.extend(notes_per_song)

    # Ordered list of unique pitches.
    pitches = sorted(set(notes))
    # print(pitches)

    # Get network inputs and outputs (labels).
    net_in_raw, net_in, net_out = generate_data_sequences(notes=notes, pitches=pitches)
    # print("Network inputs: ", net_in)
    # print("Network outputs: ", net_out)

    # Create model.
    model = create_model(net_in=net_in, pitches=pitches)
    print(model.summary())

    # Train model or load weights from file.
    if not args.no_train:
        train(model=model, x=net_in, y=net_out, batch_size=args.batch_size, epochs=args.num_epochs)
        model.save_weights(os.path.join(args.tmp_dir, args.final_weights_file))
    else:
        model.load_weights(os.path.join(args.tmp_dir, args.final_weights_file))
    print("Got trained model!")
    sys.stdout.flush()

    # Generation prediction of model.
    print("||||||||||||||||||||||||||||||||||||||| Making New Song!")
    model_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches, num_notes=args.output_notes)
    print("||||||||||||||||||||||||||||||||||||||| Done Making New Song!")
    print("Predicted Notes:", model_prediction)
    sys.stdout.flush()

    # Check similarity.
    uniqueness_score = check_uniqueness(song_bank=notes_2d, test_song=model_prediction)

    # Create output notes and save output.
    print("||||||||||||||||||||||||||||||||||||||| Generating final output!")
    output_notes = create_output_notes(prediction_data=model_prediction)

    # Save output song in the format requested by the user.
    midi_stream = stream.Stream(output_notes)
    output_file = args.output_file
    if output_file.endswith(".mid"):
        midi_stream.write("midi", fp=args.output_file)
    elif output_file.endswith(".wav"):
        tmp_output = os.path.join(args.tmp_dir, "tmp_output.mid")
        midi_stream.write("midi", fp=tmp_output)
        fs = FluidSynth()
        fs.midi_to_audio(tmp_output, output_file)
    elif output_file.endswith(".mp3"):
        tmp_output_mid = os.path.join(args.tmp_dir, "tmp_output.mid")
        tmp_output_wav = os.path.join(args.tmp_dir, "tmp_output.wav")
        midi_stream.write("midi", fp=tmp_output_mid)
        fs = FluidSynth()
        fs.midi_to_audio(tmp_output_mid, tmp_output_wav)
        sound = AudioSegment.from_wav(tmp_output_wav)
        sound.export(output_file, format="mp3")
    else:
        assert False, "Unsupported output file!"
    print("Saved prediction to: ", output_file)

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nEnd")
    sys.stdout.flush()
