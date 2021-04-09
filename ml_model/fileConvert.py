import os
import time
from pydub import AudioSegment
SPLIT_INPUT_MIDI_NAME = "input_songs/splitSongs/"

# when run from command prompt, all training songs are converted to .mid for use later in the program
# py fileConvert.py

# converts separated songs to midi
for song_folder in os.listdir(SPLIT_INPUT_MIDI_NAME):
    currentDirectory = os.listdir(SPLIT_INPUT_MIDI_NAME + song_folder)
    coreDirectory = "C:\\Users\Puppyvolcano\PycharmProjects\MusicAIAlpha\\"  # project file location
    songConvert = 0  # keeps program from sleeping when no songs are being converted
    inputFormat = "wav"
    for outputSongName in currentDirectory:
        # if wav file has not already been converted, convert it
        if inputFormat in outputSongName:
            if not currentDirectory.__contains__(outputSongName[0:outputSongName.index(inputFormat)] + "mid"):
                songConvert += 1
                newFileName = SPLIT_INPUT_MIDI_NAME + '"' + song_folder + '"' + "/" + outputSongName[0:outputSongName.index(inputFormat)]

                cmdInput = "audio-to-midi --output " + newFileName + "mid " + newFileName + "wav" + " -a 0.9 -t 500 -C 10 -s"  #

                # only shows progress for one song per batch to prevent cluttering
                #if songConvert != 1:
                #    cmdInput.append(" -n")
                print(cmdInput)
                songStream = os.popen(cmdInput)
    time.sleep(20 * songConvert)
    
    
