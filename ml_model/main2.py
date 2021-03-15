import glob
import os
import pickle
import sys
import wave
from pydub import AudioSegment
import musdb
import multiprocessing
#import audio_to_midi

from music21 import chord, converter, instrument, note, stream, common
import numpy as np
import tensorflow as tf
import spleeter
import time


# Song links: https://bitmidi.com/queen-bohemian-rhapsody-mid
#             https://bitmidi.com/toto-africa-mid


# separates all music in the train songs folder using spleeter
def separate_music():
    trainSongSource = "input_songs/trainSongs/"
    separatedSongs = os.listdir("input_songs/splitSongs")  # gets list of songs which have already been split
    print("olive")

    # loops through all songs in train songs folder
    for trainSong in os.listdir(trainSongSource):
        # gets index of file extension so songs can be filtered
        fileTypeIndex = 0
        if trainSong.__contains__(".mp3"):
            fileTypeIndex = trainSong.index(".mp3")
        elif trainSong.__contains__(".wav"):
            fileTypeIndex = trainSong.index(".wav")
        elif trainSong.__contains__(".mid"):
            fileTypeIndex = trainSong.index(".mid")

        # if song has already been split, it is skipped
        if not separatedSongs.__contains__(trainSong[0:fileTypeIndex]):
            # command for splitting input song. Currently using absolute path for storage location, hopefully a workaround
            # can be found in the future
            cmdInput = "spleeter separate " + trainSongSource + '"' + trainSong + '"' + " -p spleeter:" + str(MAX_STEMS)\
                       + "stems -o \\Users\Puppyvolcano\PycharmProjects\MusicAIAlpha\input_songs\splitSongs"# -c mp3"
            print(cmdInput)
                       # -c. E.G. -c mp3 outputs in mp3 instead of wav

            songStream = os.popen(cmdInput)
            output = songStream.read()
            print(output)
        else:
            print(trainSong + " already split")

    # converts music files to midi. Only works with Python 3.9 which some other functions don't work with.
    # currently converting using command line, multiple Python channels is our backup plan
    #convert_music()


# converts separated songs to midi
def convert_music():
    for song_folder in os.listdir(SPLIT_INPUT_MIDI_NAME):
        for outputSongName in os.listdir(SPLIT_INPUT_MIDI_NAME + song_folder):
            if "wav" in outputSongName:

                outputSong = AudioSegment.from_wav(SPLIT_INPUT_MIDI_NAME + song_folder + "/" + outputSongName)
                newFileName = SPLIT_INPUT_MIDI_NAME + song_folder + "/" + outputSongName[0:outputSongName.index(".wav")] + ".wav"
                cmdInput = "audio-to-midi " + newFileName
                print(cmdInput)
                songStream = os.popen(cmdInput)


# TODO - Convert these to command line flags.
USE_GPU = True
USE_CACHE = False
SEPARATE_MUSIC = False
TRAIN_NETWORK = True
TMP_DIR = 'tmp'

# Input music params.
CACHE_FILE = "notes"
INPUT_MIDI_FILES = glob.glob("input_songs/*.mid")
SPLIT_INPUT_MIDI_NAME = "input_songs/splitSongs/"

# Model params.
LSTM_SEQ_LENGTH = 100  # should be higher, initially 100
NUM_EPOCHS = 20  # initially 200
BATCH_SIZE = 128
TRAINING_WEIGHTS_FN = "weights_training_E{epoch:04d}_L{loss:.4f}.hdf5"
FINAL_WEIGHTS_FILE = "weights_trained" #.hdf5
OUTPUT_NOTES = 600
MAX_SONG_NOTES = 1000   # not currently used, meant to cap notes used from single song
OUTPUT_FILE = "model_output" # .mid
# number of instruments to train for
MAX_STEMS = 5
MAX_PROCESSES = 5

userLoop = False

# function for separating instruments within a song
if SEPARATE_MUSIC:
    separate_music()

# gives user option to change model parameters
#tempInput = input("Do you want to change any model parameters? y/n\n").lower()
#if tempInput == 'y':
#    userLoop = True

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


def get_note_now(fn, note_list, processNum):
    notes = []
    print("Parsing file: ", fn)

    midi = converter.parse(fn)
    elapsedTime = 0.0

    print("Finished parsing file: ", fn)

    try:  # File has instrument parts.
        s2 = instrument.partitionByInstrument(midi)
        notes_to_parse = s2.parts[0].recurse()
    except:  # File has notes in a flat structure.
        notes_to_parse = midi.flat.notes

    # sets up vars used to discard repetitive input from song. Can be an issue due to the operations
    # that have been performed on files up to this point
    lastChord = chord.Chord(['A', 'B']).normalOrder
    lastNote = note.Note('A').pitch

    sameChord = 0
    sameNote = 0
    maxRepetitions = 2

    # goes through all notes in song and adds them if they are eligible
    for ele in notes_to_parse:
        #if isinstance(ele, note.Note):
        #    print(lastNote, ele.pitch, lastNote == ele.pitch)
        #elif isinstance(ele, chord.Chord):
        #    print(lastChord, ele.normalOrder, lastChord == ele.normalOrder)

        # adds note/chord to list if they are not repeated too many times in a row
        if isinstance(ele, note.Note):
            if lastNote == ele.pitch:
                sameNote += 1
            else:
                sameNote = 0
            lastNote = ele.pitch
            if sameNote < maxRepetitions:
                notes.append(str(ele.pitch))
        elif isinstance(ele, chord.Chord):
            if lastChord == ele.normalOrder:
                sameChord += 1
            else:
                sameChord = 0
            lastChord = ele.normalOrder
            if sameChord < maxRepetitions:
                notes.append('.'.join(str(n) for n in ele.normalOrder))
        else:
            # TODO(deev): Handle Instrument Types.
            continue
        # elapsedTime += timedelta()

    # TODO(Alden): Add support for note timing in addition to pitch.
    # for n in notes:
    #    print(n)
    note_list[processNum] = [notes]
    return notes


# Get all the notes and chords from the midi files.
def get_notes_from_midi(fns):
    # multithreaded file parsing
    #noteList = common.runParallel(fns, parallelFunction=get_note_now)
    waitTime = 50  # time in seconds that the process will run in seconds before stopping

    processTime = [waitTime]
    processList = [multiprocessing.Process()]
    manager = multiprocessing.Manager()
    noteList = manager.dict()
    jobs = []

    x = 1
    # initializes lists at correct number of processes for user's machine
    while x < MAX_PROCESSES:
        processTime += [waitTime - x]
        processList += [multiprocessing.Process()]
        x += 1

    iter1 = 0
    while iter1 < len(fns):
        x = 0
        #print(processTime)
        for proc in processList:
            if processTime[x] > waitTime and iter1 < len(fns):
                if proc.is_alive():
                    proc.terminate()
                    print("process terminated")
                    proc.join()
                iter1 += 1

                fn = fns[iter1]
                processList[x] = multiprocessing.Process(target=get_note_now, name="getNote", args=(fn, noteList, iter1,))
                jobs.append(processList[x])
                processTime[x] = 0

                processList[x].start()
                iter1 += 1
            x += 1
        time.sleep(1)
        i = 0
        while i < len(processTime):
            processTime[i] += 1
            i += 1
            
    time.sleep(waitTime)

    for proc in jobs:
        if proc.is_alive():
            proc.terminate()
            print("process terminated")
        proc.join()

    midNotes = []
    allNotes = []
    #print(noteList)
    for s in noteList.values():
        for n in s:
           midNotes.append(n)

    for s in midNotes:
        for n in s:
            allNotes.append(n)

    #print(allNotes)
    return allNotes


    #note1 = []


    #for fn in fns:
    #    notes.append(get_note_now())
    #print(note)



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
    for f in glob.glob(os.path.join(str(TMP_DIR + str(currentStem)), TRAINING_WEIGHTS_FN[:TRAINING_WEIGHTS_FN.find('{')])):
        os.remove(f)

    # Define checkpoint to save weights during training.
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(str(TMP_DIR + str(currentStem)), TRAINING_WEIGHTS_FN),
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


def create_midi(prediction_data, current_stem):
    """ Convert the output from the prediction to notes and create a midi file from the notes """
    offset = 0
    output_notes = []
    currentInstrument = instrument.Piano()

    # changes stored instrument based on stem
    if current_stem == 1:
        currentInstrument = instrument.Piano()
    elif current_stem == 2:
        currentInstrument = instrument.Bass()
    elif current_stem == 3:
        currentInstrument = instrument.BassDrum()
    elif current_stem == 4:
        currentInstrument = instrument.Vocalist()
    # last stem is currently all instruments that are not sorted out
    elif current_stem == 5:
        currentInstrument = instrument.Maracas()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_data:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = currentInstrument
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset

            # changes stored instrument based on stem
            if current_stem == 1:
                new_note.storedInstrument = instrument.Piano()
            elif current_stem == 2:
                new_note.storedInstrument = instrument.Bass()
            elif current_stem == 3:
                new_note.storedInstrument = instrument.BassDrum()
            elif current_stem == 4:
                new_note.storedInstrument = instrument.Vocalist()
            # last stem is currently all instruments that are not sorted out
            elif current_stem == 5:
                new_note.storedInstrument = instrument.Maracas()

            #new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # TODO(deev): Handle Instrument Types.

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=str(OUTPUT_FILE + str(current_stem) + ".mid"))
    print("Saved prediciton to: ", str(OUTPUT_FILE + str(current_stem) + ".mid"))


if __name__ == '__main__':
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nStart")

    # Print system and device information.
    print("Python version: ", sys.version)
    print("Version info:", sys.version_info)
    print("Tensorflow version:", tf.__version__)
    print("Tensorflow Devices:", TF_DEVICES)

    currentStem = 1
    # creates model and outputs song for each instrument
    while currentStem <= MAX_STEMS:
        stemTempDir = str(TMP_DIR + str(currentStem))
        # Create directories for notes cache and weights.
        if not os.path.exists(stemTempDir):
            os.makedirs(stemTempDir)

        # gets music sample for current instrument
        if not USE_CACHE:
            tempName = ""
            # instrument processing order is piano, bass, drums, vocals, and other
            if currentStem == 1:
                tempName = "piano"
            elif currentStem == 2:
                tempName = "bass"
            elif currentStem == 3:
                tempName = "drums"
            elif currentStem == 4:
                tempName = "vocals"
            elif currentStem == 5:
                tempName = "other"

            # gets names of all relevant songs from subdirectories
            trainSongFiles = []
            for songFolder in os.listdir(SPLIT_INPUT_MIDI_NAME):
                targetLocation = SPLIT_INPUT_MIDI_NAME + songFolder + "/" + tempName + ".mid"
                trainSongFiles += glob.glob(targetLocation)
            #print(trainSongFiles)

            # import files
            notes = get_notes_from_midi(fns=trainSongFiles)
            with open(os.path.join(stemTempDir, CACHE_FILE), "wb") as f:
                pickle.dump(notes, f)
        else:
            with open(os.path.join(stemTempDir, CACHE_FILE), "rb") as f:
                notes = pickle.load(f)

                for n in notes:
                    print(n)

        """# Get notes. # Old file import
        if not USE_CACHE:
            notes = get_notes_from_midi(fns=INPUT_MIDI_FILES)
            with open(os.path.join(TMP_DIR, CACHE_FILE), "wb") as f:
                pickle.dump(notes, f)
        else:
            with open(os.path.join(TMP_DIR, CACHE_FILE), "rb") as f:
                notes = pickle.load(f)"""
        # print(notes)

        # Ordered list of unique pitches.
        pitches = sorted(set(notes))
        print(pitches)

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
            model.save_weights(os.path.join(str(TMP_DIR + str(currentStem)), (FINAL_WEIGHTS_FILE + str(currentStem) + ".hdf5")))
        else:
            model.load_weights(os.path.join(str(TMP_DIR + str(currentStem)), (FINAL_WEIGHTS_FILE + str(currentStem) + ".hdf5")))
        print("Got trained model!")

        # Generation prediction of model.
        model_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches)
        print("Predicted Notes:", model_prediction)

        # Save prediction as midi file.
        create_midi(prediction_data=model_prediction, current_stem=currentStem)
        currentStem += 1

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nEnd")
