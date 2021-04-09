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


# error that occurred during a training session
"""2021-03-30 16:24:24.394917: E tensorflow/stream_executor/cuda/cuda_event.cc:29] Error polling for event status: failed to query event: CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure
2021-03-30 16:24:24.395092: F tensorflow/core/common_runtime/gpu/gpu_event_mgr.cc:220] Unexpected Event status: 1

Process finished with exit code -1073740791 (0xC0000409)"""

# TODO - Convert these to command line flags.
USE_GPU = True
USE_CACHE = False
SEPARATE_MUSIC = False
TRAIN_NETWORK = True
TRAIN_TIMING = False  # set to true if note timing training is desired in addition to pitch
PLAY_SONG = False
TMP_DIR = 'tmp'

# Input music params.
CACHE_FILE = "notes"
INPUT_MIDI_FILES = glob.glob("input_songs/*.mid")
SPLIT_INPUT_MIDI_NAME = "input_songs/splitSongs/"

# Model params.
LSTM_SEQ_LENGTH = 100  # should be higher, initially 100
NUM_EPOCHS = 400  # initially 200
BATCH_SIZE = 128
SIMILARITY_CUTOFF = 0.5 # maximum ratio of pitches that can match a training song to be rejected (in any order)
LENGTH_CUTOFF = 40  # minimum number of valid notes for song to be used for model training
MAX_SONG_NOTES = 200  # max number of notes that will be accepted per song

# file locations for pitch weights
TRAINING_WEIGHTS_FN = "weights_training_E.hdf5"
# TRAINING_WEIGHTS_FN = "weights_training_E{epoch:04d}_L{loss:.4f}.hdf5"
FINAL_WEIGHTS_FILE = "weights_trained.hdf5"
OUTPUT_NOTES = 300
OUTPUT_FILE = "model_output" # .mid
# number of instruments to train for
MAX_STEMS = 5
MAX_PROCESSES = 4
userLoop = False


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

# keeps track of counts of all notes/chords
noteCounts = {}
totalNotes = 1
chordCounts = {}
totalChords = 1


# multiprocessing note collector
def get_note_now(fn, note_list, process_num, train_type):
    global noteCounts
    global totalNotes
    global chordCounts
    global totalChords

    collected_notes = []
    song_notes = 1

    # parses notes from sound files
    print("Parsing file: ", fn)

    midi = converter.parse(fn)

    print("Finished parsing file: ", fn)

    # will collect note pitch instead of timing
    if train_type == "pitch":
        try:  # File has instrument parts.
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # File has notes in a flat structure.
            notes_to_parse = midi.flat.notes

        # prints duration for each note
        #for m in notes_to_parse:
        #    print(m.duration.quarterLength)

        # sets up vars used to discard repetitive input from song. Can be an issue due to the operations
        # that have been performed on files up to this point
        lastChord = ""  # chord.Chord(['A', 'B']).normalOrder""
        lastNote = ""  # note.Note('A').pitch

        # starting vars for chords / notes, max repetitions allowed by default to filter out noise
        sameChord = 0
        sameNote = 0
        maxRepetitions = 2

        # max portion of the song a note can take up to be added
        noteFrequencyCutoff = 0.15  # 0.2
        chordFrequencyCutoff = 0.25  # 0.35
        starterNotes = 10   # allows repetitive notes at the start of song to give time for variation to even out

        # goes through all notes in song and adds them if they are eligible
        for ele in notes_to_parse:
            # stops parsing notes once the desired number has been reached
            if song_notes > MAX_SONG_NOTES:
                if len(collected_notes) > LENGTH_CUTOFF:
                    note_list[process_num] = [collected_notes]
                    return collected_notes
                else:
                    note_list[process_num] = [[]]
                    return []

            # adds note/chord to list if they are not repeated too many times in a row
            if isinstance(ele, note.Note):
                # adds note to dictionary if it is not there
                if str(ele.pitch) not in noteCounts:
                    noteCounts[str(ele.pitch)] = 0

                # if note has not repeated too much and it isn't dominating the song, add it to collected notes
                if noteCounts[str(ele.pitch)] / totalNotes < noteFrequencyCutoff or totalNotes < starterNotes:
                    # tracks repeated occurrences of notes
                    if lastNote == str(ele.pitch):
                        sameNote += 1
                    else:
                        sameNote = 0
                    lastNote = str(ele.pitch)

                    if sameNote < maxRepetitions:
                        totalNotes += 1
                        song_notes += 1

                        # adds note to list and increments dictionary entry
                        collected_notes.append(str(ele.pitch))
                        noteCounts[str(ele.pitch)] = noteCounts[str(ele.pitch)] + 1

                        # if note is different than last accepted note, reset counter
                        if lastNote != str(ele.pitch):
                            sameNote = 0

            elif isinstance(ele, chord.Chord):
                totalChords += 1

                tempChord = ""
                # checks individual notes of chord to make sure they are not too common
                for n in ele.normalOrder:
                    # adds chord to dictionary if it is not there
                    if str(n) not in chordCounts:
                        chordCounts[str(n)] = 0
                    if chordCounts[str(n)] / totalChords < chordFrequencyCutoff or totalChords < starterNotes:
                        chordCounts[str(n)] = chordCounts[str(n)] + 1
                    tempChord += '.' + str(n)

                # formats chord
                tempChord = tempChord[1:len(tempChord)]

                if str(tempChord) not in chordCounts:
                    chordCounts[str(tempChord)] = 0

                if lastChord == tempChord:
                    sameChord += 1
                else:
                    sameChord = 0

                # if chord has not repeated too much and it isn't dominating the song, add it to collected notes
                if sameChord < maxRepetitions:# and (chordCounts[str(tempChord)] / totalChords < chordFrequencyCutoff or totalChords < starterNotes):
                    # adds chord to list and increments dictionary entry
                    if tempChord != "":
                        collected_notes.append(tempChord)
                        song_notes += 1
                    else:
                        totalChords -= 1

                lastChord = tempChord
            else:
                #print(ele)
                # TODO(deev): Handle Instrument Types.
                continue

    # will collect note timing instead of pitch
    elif train_type == "timing":
        try:  # File has instrument parts.
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:  # File has notes in a flat structure.
            notes_to_parse = midi.flat.notes

        # sets up vars used to discard repetitive input from song. Can be an issue due to the operations
        # that have been performed on files up to this point
        lastNote = note.Note('A').duration.quarterLength
        lastChord = chord.Chord(['A', 'B']).duration.quarterLength

        # starting vars for chords / notes, max repetitions allowed by default to filter out noise
        sameChord = 0
        sameNote = 0
        maxRepetitions = 15

        # goes through all notes in song and adds them if they are eligible
        for ele in notes_to_parse:
            # adds note/chord to list if they are not repeated too many times in a row
            if isinstance(ele, note.Note):
                if lastNote == ele.duration.quarterLength:
                    sameNote += 1
                else:
                    sameNote = 0
                lastNote = ele.pitch
                if sameNote < maxRepetitions:
                    collected_notes.append(str(ele.duration.quarterLength))
            elif isinstance(ele, chord.Chord):
                if lastChord == ele.duration.quarterLength:
                    sameChord += 1
                else:
                    sameChord = 0
                lastChord = ele.duration.quarterLength
                if sameChord < maxRepetitions:
                    collected_notes.append(str(ele.duration.quarterLength))
                    #collected_notes.append('.'.join(str(n) for n in ele.duration.quarterLength))
            else:
                continue

    # returns notes if they meet the minimum cutoff
    if len(collected_notes) > LENGTH_CUTOFF:
        note_list[process_num] = [collected_notes]
        return collected_notes
    else:
        note_list[process_num] = [[]]
        return []


# Get all the notes and chords from the midi files.
def get_notes_from_midi(fns, train_type):
    # multithreaded file parsing
    # noteList = common.runParallel(fns, parallelFunction=get_note_now) # old multiprocessing method
    waitTime = 50  # time in seconds that the process will run for before stopping job

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
    # adds and removes jobs until all songs have begun parsing
    while iter1 < len(fns):
        x = 0
        #print(processTime)
        i = 0
        while i < len(processTime):
            if (processTime[i] > waitTime or not processList[i].is_alive()) and iter1 < len(fns):
                if processList[i].is_alive():
                    processList[i].terminate()
                    print("process terminated")
                    processList[i].join()
                #iter1 += 1

                #if iter1 < len(fns):
                fn = fns[iter1]
                processList[i] = multiprocessing.Process(target=get_note_now, name="getNote", args=(fn, noteList, iter1, train_type,))
                jobs.append(processList[i])
                processTime[i] = 0

                processList[i].start()
                iter1 += 1
                x += 1
            i += 1
        time.sleep(1)
        i = 0
        while i < len(processTime):
            processTime[i] += 1
            i += 1

    print("Waiting for song parsing to finish...")
    time.sleep(waitTime)

    # remove all remaining jobs after waiting necessary time
    for proc in jobs:
        if proc.is_alive():
            proc.terminate()
            print("process terminated")
        proc.join()

    # converts note list into easier to use format then returns it
    allNotes = []
    # print(noteList)
    for s in noteList.values():
        for nt in s:
            allNotes.append(nt)

    print("All notes: " + str(allNotes))
    return allNotes

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
    for f in glob.glob(os.path.join(stemTempDir, TRAINING_WEIGHTS_FN[:TRAINING_WEIGHTS_FN.find('{')])):
        os.remove(f)

    # Define checkpoint to save weights during training.
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(stemTempDir, TRAINING_WEIGHTS_FN),
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


# loops through all songs and generates uniqueness score for each. Returns worst performance, so the similarity to the
# song that is closest to the test song
def check_uniqueness(song_bank, test_song):
    # disregards songs that are too short to impactfully compare
    # tracks number and type of notes for test song
    check_notes = 0
    check_counts = {}

    # finds frequency of all notes in generated song
    for songNote in test_song:
        if songNote not in check_counts:
            check_counts[songNote] = 0

        check_counts[songNote] = check_counts[songNote] + 1
        check_notes += 1

    worstScore = 0.0
    # goes through each song that was used for training and compares each note frequency to the generated song
    # keeps the score of the song that is most similar to the generated song by this metric, which will be
    # used for filtering later
    for trainSong in song_bank:
        # tracks number and type of notes for train song
        test_counts = {}
        test_notes = 0

        # finds frequency of all notes in generated song
        for songNote in trainSong:
            if songNote not in test_counts:
                test_counts[songNote] = 0

            test_counts[songNote] = test_counts[songNote] + 1
            test_notes += 1

        sim_score = 1.0
        # compare frequencies of notes
        if len(trainSong) > LENGTH_CUTOFF:
            for noteType in test_counts:
                if noteType in check_counts:
                    sim_score -= abs(test_counts[noteType] / test_notes - check_counts[noteType] / check_notes)
                else:
                    sim_score -= test_counts[noteType] / test_notes
        else:
            sim_score = 0.0
        if sim_score > worstScore:
            worstScore = sim_score

    print("Similarity Score: ", str(worstScore))

    return worstScore


# Convert the output from the prediction to notes and create a midi file from the notes
def create_midi(prediction_data, current_stem):
    offset = 0
    output_notes = []
    currentInstrument = instrument.Piano()

    # changes stored instrument based on stem
    if current_stem == 1:
        currentInstrument = instrument.Piano()
    elif current_stem == 2:
        currentInstrument = instrument.BassGuitar()
    elif current_stem == 3:
        currentInstrument = instrument.SteelDrum()
    elif current_stem == 4:
        currentInstrument = instrument.Whistle()
    # last stem is currently all instruments that are not sorted out
    elif current_stem == 5:
        currentInstrument = instrument.Guitar()

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_data:
        # Pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []

            # converts notes in chord to music 21 notes
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = currentInstrument
                notes.append(new_note)

            # saves chord as desired instrument
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(currentInstrument)
            output_notes.append(new_chord)

        # Pattern is a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset

            # changes stored instrument based on stem
            new_note.storedInstrument = currentInstrument

            output_notes.append(currentInstrument)
            output_notes.append(new_note)

        # TODO(deev): Handle Instrument Types.

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=str(OUTPUT_FILE + str(current_stem) + ".mid"))
    if PLAY_SONG:
        midi_stream.show('midi')  # plays midi file after generation is finished
    print("Saved prediciton to: ", str(OUTPUT_FILE + str(current_stem) + ".mid"))


if __name__ == '__main__':
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nStart")

    # Print system and device information.
    print("Python version: ", sys.version)
    print("Version info:", sys.version_info)
    print("Tensorflow version:", tf.__version__)
    print("Tensorflow Devices:", TF_DEVICES)

    currentStem = 2
    # creates model and outputs song for each instrument
    while currentStem <= MAX_STEMS:
        # PITCH
        # Model is first trained to learn pitch
        trainingType = "pitch"
        noteBank = []
        notes = []
        
        # used to make subdirectories and direct training based on the desired parameter to train for
        stemTempDir = str(TMP_DIR + str(currentStem) + trainingType)
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
            # print(trainSongFiles)

            # import files
            notes = []
            noteBank = get_notes_from_midi(fns=trainSongFiles, train_type=trainingType)

            for s in noteBank:
                for nt in s:
                    notes.append(nt)
            with open(os.path.join(stemTempDir, CACHE_FILE), "wb") as f:
                pickle.dump(noteBank, f)
        else:
            with open(os.path.join(stemTempDir, CACHE_FILE), "rb") as f:
                noteBank = pickle.load(f)
            for s in noteBank:
                for nt in s:
                    notes.append(nt)

        #print(notes)

        # Ordered list of unique pitches.
        pitches = sorted(set(notes))
        #print(pitches)

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
            model.save_weights(os.path.join(stemTempDir, FINAL_WEIGHTS_FILE))
        else:
            model.load_weights(os.path.join(stemTempDir, FINAL_WEIGHTS_FILE))
        print("Got trained model!")

        similarityScore = 1.0
        model_prediction = []

        while similarityScore > SIMILARITY_CUTOFF:
            # Generation prediction of model.
            model_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches)
            print("Predicted Notes:", model_prediction)
            similarityScore = check_uniqueness(noteBank, model_prediction)

        # Save prediction as midi file.
        create_midi(prediction_data=model_prediction, current_stem=currentStem)
        currentStem += 1

        # TIMING
        # model is trained again on the same songs based on timing instead of pitch.
        # The results are combined and output as one song
        if TRAIN_TIMING:
            trainingType = "timing"

            stemTempDir = str(TMP_DIR + str(currentStem) + trainingType)
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
                # print(trainSongFiles)

                # import files
                notes = []
                noteBank = get_notes_from_midi(fns=trainSongFiles, train_type=trainingType)

                for s in noteBank:
                    for nt in s:
                        notes.append(nt)
                with open(os.path.join(stemTempDir, CACHE_FILE), "wb") as f:
                    pickle.dump(noteBank, f)
            else:
                with open(os.path.join(stemTempDir, CACHE_FILE), "rb") as f:
                    noteBank = pickle.load(f)
                for s in noteBank:
                    for nt in s:
                        notes.append(nt)

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
                model.save_weights(os.path.join(stemTempDir, FINAL_WEIGHTS_FILE))
            else:
                model.load_weights(os.path.join(stemTempDir, FINAL_WEIGHTS_FILE ))
            print("Got trained model!")

            similarityScore = 1.0
            model_prediction = []

            while similarityScore > SIMILARITY_CUTOFF:
                # Generation prediction of model.
                model_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches)
                print("Predicted Notes:", model_prediction)
                similarityScore = check_uniqueness(noteBank, model_prediction)

            # Save prediction as midi file.
            #create_midi(prediction_data=model_prediction, current_stem=currentStem)
            #currentStem += 1

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nEnd")
