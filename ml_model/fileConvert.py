import os
import time
from pydub import AudioSegment
SPLIT_INPUT_MIDI_NAME = "input_songs/splitSongs/"

# when run from command prompt, all training songs are converted to .mid for use later in the program

# converts separated songs to midi
for song_folder in os.listdir(SPLIT_INPUT_MIDI_NAME):
    currentDirectory = os.listdir(SPLIT_INPUT_MIDI_NAME + song_folder)
    coreDirectory = "C:\\Users\Puppyvolcano\PycharmProjects\MusicAIAlpha\\"  # project file location
    songConvert = 0  # keeps program from sleeping when no songs are being converted
    inputFormat = "mp3"
    for outputSongName in currentDirectory:
        # if wav file has not already been converted, convert it
        if inputFormat in outputSongName:
            if not currentDirectory.__contains__(outputSongName[0:outputSongName.index(inputFormat)] + "mid"):
                songConvert += 1
                newFileName = SPLIT_INPUT_MIDI_NAME + '"' + song_folder + '"' + "/" + outputSongName[0:outputSongName.index(inputFormat)] + "mid"

                sound = AudioSegment.from_mp3(coreDirectory + SPLIT_INPUT_MIDI_NAME + '"' + song_folder + '"' + "/" + outputSongName[0:outputSongName.index(inputFormat)] + "mp3")
                sound.export(coreDirectory + SPLIT_INPUT_MIDI_NAME + '"' + song_folder + '"' + "/" + outputSongName[0:outputSongName.index(inputFormat)] + "wav", format="wav")

                cmdInput = "audio-to-midi --output " + newFileName + SPLIT_INPUT_MIDI_NAME + '"' + song_folder + '"' + "/" + outputSongName[0:outputSongName.index(inputFormat)] + "mp3"
                print(cmdInput)
                print(song_folder)
                #songStream = os.popen(cmdInput)
    time.sleep(20 * songConvert)

