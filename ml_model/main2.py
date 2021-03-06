import glob
import os
import pickle
import sys
from pydub import AudioSegment
import multiprocessing

from music21 import chord, converter, instrument, note, stream, common
import numpy as np
import tensorflow as tf
import time
import argparse

# initializes parser so constants can be changed more easily
parser = argparse.ArgumentParser()

# Song links: testing songs were taken from https://www.mutopiaproject.org/

parser.add_argument("--USE_GPU", type=bool, default=True, help="use GPU to speed up training")
parser.add_argument("--USE_CACHE", type=bool, default=True, help="use note cache or parse new songs")
parser.add_argument("--SEPARATE_MUSIC", type=bool, default=False, help="separate music or use splitSongs directory")
parser.add_argument("--TRAIN_NETWORK", type=bool, default=False, help="train network or use generated weights")
parser.add_argument("--PLAY_SONG", type=bool, default=False, help="choose whether to play completed song")
parser.add_argument("--TMP_DIR", type=str, default="tmp", help="directory to store weights and cache")


# Input music params.
parser.add_argument("--CACHE_FILE", type=str, default="notes", help="file location to store cached notes")
parser.add_argument(
    "--SPLIT_INPUT_MIDI_NAME", type=str, default="input_songs/splitSongs/", help="location split songs are stored"
)
parser.add_argument(
    "--TRAIN_SONG_SOURCE",
    type=str,
    default="input_songs/trainSongs/",
    help="location training songs are pulled from (will be split)",
)

parser.add_argument(
    "--SPLIT_SONGS_DIR",
    type=str,
    default="input_songs/splitSongs",
    help="storage location for splitting songs)",
)

parser.add_argument("--LSTM_SEQ_LENGTH", type=int, default=20, help="sequence length modifier for notes")
parser.add_argument("--NUM_EPOCHS", type=int, default=250, help="number of training generations")
parser.add_argument("--NUM_TIME_EPOCHS", type=int, default=200, help="number of training generations for timing")
parser.add_argument("--BATCH_SIZE", type=int, default=128, help="batch size for model")
parser.add_argument(
    "--SIMILARITY_CUTOFF",
    type=float,
    default=0.6,
    help="maximum ratio of pitches that can match a training song to be rejected (in any order)",
)
parser.add_argument(
    "--LENGTH_CUTOFF", type=int, default=40, help="minimum number of valid notes for song to be used for model training"
)
parser.add_argument("--MAX_SONG_NOTES", type=int, default=400, help="max number of notes that will be accepted per song")

# file locations for pitch weights
parser.add_argument(
    "--TRAINING_WEIGHTS_FN", type=str, default="weights_training_E.hdf5", help="name of training weight file"
)
parser.add_argument("--FINAL_WEIGHTS_FILE", type=str, default="weights_trained.hdf5", help="name of final weight file")
parser.add_argument("--OUTPUT_NOTES", type=int, default=100, help="number of notes in output file")
parser.add_argument("--OUTPUT_FILE", type=str, default="model_output", help="output file location")
parser.add_argument("--MAX_STEMS", type=int, default=5, help="number of instruments to train for")
parser.add_argument("--MAX_PROCESSES", type=int, default=4, help="max number of parsing processes")

# used to override train_network and use_cache so they only apply to timing or training. If false,
# note cache and the most recent pretrained model will be used. Main purpose is to make testing easier
parser.add_argument("--TRAIN_TIMING", type=bool, default=True, help="set to true if note timing training is desired")
parser.add_argument("--TRAIN_PITCH", type=bool, default=True, help="set to true if note pitch training is desired")
# Input music params.
parser.add_argument("--RAW_MIDI_LOC", type=list,
                    default=["input_songs/tempPiano/", "input_songs/tempBass/", "input_songs/tempDrums/",
                             "input_songs/tempVocals/", "input_songs/tempMisc/"],
                    help="location raw midi files for each instrument are stored")

parser.add_argument("--IMPORT_FROM_MIDI", type=bool, default=True,
                    help="determines whether songs need to be split or are midi files that can be read directly")

# adds arguments to parser
args = parser.parse_args()


# separates all music in the train songs folder using spleeter
def separate_music():
    separatedSongs = os.listdir(args.SPLIT_INPUT_MIDI_NAME)  # gets list of songs which have already been split
    print("olive")

    # loops through all songs in train songs folder
    for trainSong in os.listdir(args.TRAIN_SONG_SOURCE):
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
            cmdInput = "spleeter separate " + args.TRAIN_SONG_SOURCE + '"' + trainSong + '"' + " -p spleeter:" + str(args.MAX_STEMS)\
                       + "stems -o \\Users\Puppyvolcano\PycharmProjects\MusicAIAlpha\input_songs\splitSongs"# -c mp3"
            print(cmdInput)
                       # -c. E.G. -c mp3 outputs in mp3 instead of wav

            songStream = os.popen(cmdInput)
            output = songStream.read()
            print(output)
        else:
            print(trainSong + " already split")

    # converts music files to midi. Only works with Python 3.9 which some other functions don't work with.
    # currently converting using command line locally to test, converted with multiple Python channels on our website
    # convert_music()


# converts separated songs to midi
def convert_music():
    # goes through folder for each song
    for song_folder in os.listdir(args.SPLIT_INPUT_MIDI_NAME):
        for outputSongName in os.listdir(args.SPLIT_INPUT_MIDI_NAME + song_folder):
            if "wav" in outputSongName:
                outputSong = AudioSegment.from_wav(args.SPLIT_INPUT_MIDI_NAME + song_folder + "/" + outputSongName)
                newFileName = args.SPLIT_INPUT_MIDI_NAME + song_folder + "/" + outputSongName[0:outputSongName.index(".wav")] + ".wav"
                cmdInput = "audio-to-midi " + newFileName
                print(cmdInput)
                songStream = os.popen(cmdInput)


# function for separating instruments within a song
if args.SEPARATE_MUSIC:
    separate_music()

if not args.USE_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
TF_DEVICES = tf.config.list_physical_devices()

# keeps track of counts of all notes/chords
noteCounts = {}
totalNotes = 1
chordCounts = {}
totalChords = 1


# collects notes of desired type from given file
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

    # parse notes from split song directory
    if not args.IMPORT_FROM_MIDI:
        # will collect note pitch instead of timing
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
            if song_notes > args.MAX_SONG_NOTES:
                if len(collected_notes) > args.LENGTH_CUTOFF:
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
                        if train_type == "pitch":
                            collected_notes.append(str(ele.pitch))
                        else:
                            collected_notes.append(float(ele.duration.quarterLength))
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
                        # appends chord notes or quarter length depending on what is wanted
                        if train_type == "pitch":
                            collected_notes.append(tempChord)
                        else:
                            collected_notes.append(float(ele.duration.quarterLength))
                        song_notes += 1
                    else:
                        totalChords -= 1

                lastChord = tempChord
            else:
                continue
    # parse notes from raw midi directory, no filter needed
    else:
        try:  # File has instrument parts.
            # take part with highest note total
            s2 = instrument.partitionByInstrument(midi)

            maxPart = 0
            maxNotes = 0

            # finds part with most notes
            for parts in s2.parts:
                currentPart = 0
                currentNotes = 0
                for nts in parts.recurse():
                    currentNotes += 1
                if currentNotes > maxNotes:
                    maxNotes = currentNotes
                    maxPart = currentPart
                currentPart += 1

            #notes_to_parse = s2.parts[maxPart].recurse()
            notes_to_parse = midi.flat.notes
        except:  # File has notes in a flat structure.
            notes_to_parse = midi.flat.notes
            #print(notes_to_parse)

        # goes through all notes in song and adds them if they are eligible
        for ele in notes_to_parse:
            #print(ele)
            # stops parsing notes once the desired number has been reached
            if song_notes > args.MAX_SONG_NOTES:
                if len(collected_notes) > args.LENGTH_CUTOFF:
                    note_list[process_num] = [collected_notes]
                    return collected_notes
                else:
                    note_list[process_num] = [[]]
                    return []

            # adds note/chord to list if they are not repeated too many times in a row
            if isinstance(ele, note.Note):
                # adds note to list and increments dictionary entry
                if train_type == "pitch":
                    collected_notes.append(str(ele.pitch))
                else:
                    collected_notes.append(float(ele.duration.quarterLength))

                song_notes += 1;

            elif isinstance(ele, chord.Chord):
                tempChord = ""
                # checks individual notes of chord to make sure they are not too common
                for n in ele.normalOrder:
                    tempChord += '.' + str(n)

                # formats chord
                tempChord = tempChord[1:len(tempChord)]

                # appends chord notes or quarter length depending on what is wanted
                if train_type == "pitch":
                    collected_notes.append(tempChord)
                else:
                    collected_notes.append(float(ele.duration.quarterLength))
                song_notes += 1
            else:
                #print(ele)
                continue

    #print(collected_notes)

    # returns notes if they meet the minimum cutoff
    if len(collected_notes) > args.LENGTH_CUTOFF:
        note_list[process_num] = [collected_notes]
        return collected_notes
    else:
        note_list[process_num] = [[]]
        return []


# Iterates through all music files and calls get_note_now to parse them for notes. Calls in a multithreaded manner and
# caps parsing time for each song to speed up performance
def get_notes_from_midi(fns, train_type):
    # multithreaded file parsing
    # noteList = common.runParallel(fns, parallelFunction=get_note_now) # old multiprocessing method
    waitTime = 20  # time in seconds that the process will run for before stopping job

    processTime = [waitTime]
    processList = [multiprocessing.Process()]
    manager = multiprocessing.Manager()
    noteList = manager.dict()
    jobs = []

    x = 1
    # initializes lists at correct number of processes for user's machine
    while x < args.MAX_PROCESSES:
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


# Generates the data sequences used by the Neural Network.
# Returns (network inputs raw (unscaled), network inputs, network outputs or labels)
def generate_data_sequences(notes, pitches):
    # Map between notes and integers.
    note_to_int = {
        note: num for num, note in enumerate(pitches)
    }

    # Create raw inputs and outputs based on sequences.
    net_in_raw, net_out_raw = [], []
    for i in range(len(notes) - args.LSTM_SEQ_LENGTH):
        seq_in = notes[i: i + args.LSTM_SEQ_LENGTH]
        seq_out = notes[i + args.LSTM_SEQ_LENGTH]

        net_in_raw.append(np.array([note_to_int[note] for note in seq_in], dtype=np.float))
        net_out_raw.append(note_to_int[seq_out])

    # Reshape and normalize the input into a format compatible with LSTM layers.
    net_in = np.reshape(net_in_raw, (len(net_in_raw), args.LSTM_SEQ_LENGTH, 1))
    net_in /= len(pitches)

    # Categorize outputs.
    net_out = tf.keras.utils.to_categorical(net_out_raw)

    return net_in_raw, net_in, net_out


# creates a tensorflow LSTM based model based on information parsed from songs
def create_model(net_in, pitches):
    # initializes model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        512,
        input_shape=(net_in.shape[1], net_in.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    # adds layers
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
    # compiles model before returning it
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


# train the model for the desired number of epochs
def train(model, x, y, batch_size, epochs):
    # Delete any older weights files.
    for f in glob.glob(os.path.join(stemTempDir, args.TRAINING_WEIGHTS_FN[:args.TRAINING_WEIGHTS_FN.find('{')])):
        os.remove(f)

    # Define checkpoint to save weights during training.
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(stemTempDir, args.TRAINING_WEIGHTS_FN),
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min',
    )

    # Train model.
    model.fit(x=x, y=y, epochs=epochs, batch_size=batch_size, callbacks=[ckpt])


# uses trained model to generate output notes
def predict(model, network_in_raw, pitches, training_type):
    # Pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_in_raw)-1)

    # Map between notes and integers.
    int_to_note = {
        num: note for num, note in enumerate(pitches)
    }

    pattern = list(network_in_raw[start])
    model_out = []

    # Generate notes.
    for ni in range(args.OUTPUT_NOTES):
        # prints output every 5 notes
        if ni % 5 == 0:
            print("Generating note: ", ni)
        net_in = np.reshape(pattern, (1, len(pattern), 1))
        net_in = net_in / float(len(pitches))

        y_hat = model.predict(net_in, verbose=0)

        i = np.argmax(y_hat)

        # prints every 5th note
        # if ni % 5 == 0:
        #    print(pitches[i])

        # appends in different way depending on if pitch or timing is being trained
        if training_type == "pitch":
            model_out.append(int_to_note[i])
        else:
            model_out.append(pitches[i])

        pattern.append(i)
        pattern.pop(0)

    # returns generated song
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

    # keeps track of worst similarity score so the comparison to the most similar song is returned
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
        if len(trainSong) > args.LENGTH_CUTOFF:
            for noteType in test_counts:
                if noteType in check_counts:
                    sim_score -= abs(test_counts[noteType] / test_notes - check_counts[noteType] / check_notes)
                else:
                    sim_score -= test_counts[noteType] / test_notes
        else:
            sim_score = 0.0

        # stores current similarity score if it is the worst so far
        if sim_score > worstScore:
            worstScore = sim_score

    # prints similarity score before returning it
    print("Similarity Score: ", str(worstScore))

    return worstScore


# Convert the output from the prediction to notes and create a midi file from the notes
def create_midi(prediction_data, timing_data, current_stem):
    offset = 0
    output_notes = []
    currentInstrument = instrument.Piano()

    # changes stored instrument based on stem
    if current_stem == 1:
        currentInstrument = instrument.Piano()
    elif current_stem == 2:
        currentInstrument = instrument.ElectricBass()     #Bass()
    elif current_stem == 3:
        currentInstrument = instrument.SteelDrum()
    elif current_stem == 4:
        currentInstrument = instrument.Whistle()
    # last stem is currently all instruments that are not sorted out
    elif current_stem == 5:
        currentInstrument = instrument.Guitar()

    currentNote = 0

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

            # creates chord with predicted pitch and timing
            new_chord = chord.Chord(notes)
            new_chord.offset = offset

            # appends to list
            output_notes.append(currentInstrument)
            output_notes.append(new_chord)

        # Pattern is a note
        else:
            # creates new note
            new_note = note.Note(pattern)
            new_note.offset = offset

            # changes stored instrument based on stem
            new_note.storedInstrument = currentInstrument

            output_notes.append(currentInstrument)
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack
        # uses timing training data if available
        if len(timing_data) > currentNote:
            offset += float(timing_data[currentNote])
        else:
            offset += 0.5

        # increments current note number
        currentNote += 1

    # creates midi stream
    midi_stream = stream.Stream(output_notes)

    # writes song to file
    midi_stream.write('midi', fp=str(args.OUTPUT_FILE + str(current_stem) + ".mid"))

    # plays midi file after generation is finished if desired
    if args.PLAY_SONG:
        midi_stream.show('midi')

    # prints song storage location
    print("Saved prediciton to: ", str(args.OUTPUT_FILE + str(current_stem) + ".mid"))


# calls other functions to train model and generate songs
if __name__ == '__main__':
    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nStart")

    # Print system and device information.
    print("Python version: ", sys.version)
    print("Version info:", sys.version_info)
    print("Tensorflow version:", tf.__version__)
    print("Tensorflow Devices:", TF_DEVICES)

    # arrays used for notes
    note_prediction_bank = []
    time_prediction_bank = []

    currentStem = 1
    # creates model and outputs song for each instrument
    while currentStem <= args.MAX_STEMS:
        # PITCH
        # Model is first trained to learn pitch
        trainingType = "pitch"
        noteBank = []
        notes = []

        quarterLengths = []
        timingBank = []

        # used to make subdirectories and direct training based on the desired parameter to train for
        stemTempDir = str(args.TMP_DIR + str(currentStem) + trainingType)
        # Create directories for notes cache and weights.
        if not os.path.exists(stemTempDir):
            os.makedirs(stemTempDir)

        # gets music sample for current instrument
        if not args.USE_CACHE and args.TRAIN_PITCH:
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
            if not args.IMPORT_FROM_MIDI:
                for songFolder in os.listdir(args.SPLIT_INPUT_MIDI_NAME):
                    targetLocation = args.SPLIT_INPUT_MIDI_NAME + songFolder + "/" + tempName + ".mid"
                    trainSongFiles += glob.glob(targetLocation)
            else:
                for inputSong in os.listdir(args.RAW_MIDI_LOC[currentStem-1]):
                    trainSongFiles += [str(args.RAW_MIDI_LOC[currentStem-1] + inputSong)]
            # print(trainSongFiles)

            # import files
            notes = []
            noteBank = get_notes_from_midi(fns=trainSongFiles, train_type=trainingType)

            for s in noteBank:
                for nt in s:
                    notes.append(nt)
            with open(os.path.join(stemTempDir, args.CACHE_FILE), "wb") as f:
                pickle.dump(noteBank, f)
        # get notes from cache instead of parsing files
        else:
            with open(os.path.join(stemTempDir, args.CACHE_FILE), "rb") as f:
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
        if args.TRAIN_NETWORK and args.TRAIN_PITCH:
            train(model=model, x=net_in, y=net_out, batch_size=args.BATCH_SIZE, epochs=args.NUM_EPOCHS)
            model.save_weights(os.path.join(stemTempDir, args.FINAL_WEIGHTS_FILE))
        else:
            model.load_weights(os.path.join(stemTempDir, args.FINAL_WEIGHTS_FILE))
        print("Got trained model!")

        similarityScore = 1.0
        note_prediction = []

        # generates notes until the output song falls within the similarity cutoff
        while similarityScore > args.SIMILARITY_CUTOFF:
            # Generation prediction of model.
            note_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=pitches, training_type=trainingType)
            print("Predicted Notes:", note_prediction)
            similarityScore = check_uniqueness(noteBank, note_prediction)

        # TIMING
        # model is trained again on the same songs based on timing instead of pitch.
        # The results are combined and output as one song
        trainingType = "timing"

        stemTempDir = str(args.TMP_DIR + str(currentStem) + trainingType)
        # Create directories for notes cache and weights.
        if not os.path.exists(stemTempDir):
            os.makedirs(stemTempDir)

        # gets music sample for current instrument
        if not args.USE_CACHE and args.TRAIN_TIMING:
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
            if not args.IMPORT_FROM_MIDI:
                for songFolder in os.listdir(args.SPLIT_INPUT_MIDI_NAME):
                    targetLocation = args.SPLIT_INPUT_MIDI_NAME + songFolder + "/" + tempName + ".mid"
                    trainSongFiles += glob.glob(targetLocation)
            else:
                for inputSong in os.listdir(args.RAW_MIDI_LOC[currentStem - 1]):
                    trainSongFiles += [str(args.RAW_MIDI_LOC[currentStem - 1] + inputSong)]

            # import files
            quarterLengths = []
            timingBank = get_notes_from_midi(fns=trainSongFiles, train_type=trainingType)

            for s in timingBank:
                for nt in s:
                    quarterLengths.append(nt)
            with open(os.path.join(stemTempDir, args.CACHE_FILE), "wb") as f:
                pickle.dump(timingBank, f)
        # get notes from cache instead of parsing files
        else:
            with open(os.path.join(stemTempDir, args.CACHE_FILE), "rb") as f:
                timingBank = pickle.load(f)
            for s in timingBank:
                for nt in s:
                    quarterLengths.append(nt)

        # Ordered list of unique quarter lengths.
        noteLengths = sorted(set(quarterLengths))
        print(noteLengths)

        # Get network inputs and outputs (labels).
        net_in_raw, net_in, net_out = generate_data_sequences(notes=quarterLengths, pitches=noteLengths)

        # prints network inputs and outputs
        #print(net_out)
        #print("Network inputs: ", net_in)
        #print("Network ouputs: ", net_out)

        # Create model.
        model = create_model(net_in=net_in, pitches=noteLengths)
        print(model.summary())

        # Train model or load weights from file.
        if args.TRAIN_NETWORK and args.TRAIN_TIMING:
            train(model=model, x=net_in, y=net_out, batch_size=args.BATCH_SIZE, epochs=args.NUM_TIME_EPOCHS)
            model.save_weights(os.path.join(stemTempDir, args.FINAL_WEIGHTS_FILE))
        else:
            model.load_weights(os.path.join(stemTempDir, args.FINAL_WEIGHTS_FILE ))
        print("Got trained model!")

        similarityScore = 1.0
        timing_prediction = []

        # checks similarity of timing to songs, not currently used due to small pool of note length possibilities
        # while similarityScore > args.SIMILARITY_CUTOFF:
        # Generation prediction of model.
        timing_prediction = predict(model=model, network_in_raw=net_in_raw, pitches=noteLengths, training_type=trainingType)
        print("Predicted Quarter Lengths:", timing_prediction)
        # similarityScore = check_uniqueness(timingBank, timing_prediction)

        # adds notes to combined bank for later combined song
        note_prediction_bank += [note_prediction]
        time_prediction_bank += [timing_prediction]

        # Save prediction as midi file.
        create_midi(prediction_data=note_prediction, timing_data=timing_prediction, current_stem=currentStem)
        currentStem += 1

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\nEnd")
