# Requires python 3.9

import argparse
import os
import time

from pydub import AudioSegment

parser = argparse.ArgumentParser(description="INput sour")
parser.add_argument("--input_dir", type=str, default="input_songs/splitSongs/", help="input directory")

args = parser.parse_args()

# when run from command prompt, all training songs are converted to .mid for use later in the program
# py fileConvert.py

# converts separated songs to midi
for song_folder in os.listdir(args.input_dir):
    currentDirectory = os.listdir(args.input_dir + song_folder)
    coreDirectory = "C:\\Users\Puppyvolcano\PycharmProjects\MusicAIAlpha\\"  # project file location
    songConvert = 0  # keeps program from sleeping when no songs are being converted
    inputFormat = "wav"
    for outputSongName in currentDirectory:
        # if wav file has not already been converted, convert it
        if inputFormat in outputSongName:
            if not currentDirectory.__contains__(outputSongName[0 : outputSongName.index(inputFormat)] + "mid"):
                songConvert += 1
                newFileName = (
                    args.input_dir + '"' + song_folder + '"' + "/" + outputSongName[0 : outputSongName.index(inputFormat)]
                )

                cmdInput = "audio-to-midi --output " + newFileName + "mid " + newFileName + "wav" + " -a 0.9 -t 500 -C 10 -s"

                # only shows progress for one song per batch to prevent cluttering
                # if songConvert != 1:
                #    cmdInput.append(" -n")
                print(cmdInput)
                songStream = os.popen(cmdInput)
    time.sleep(20 * songConvert)
