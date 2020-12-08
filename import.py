import tkinter
from tkinter import filedialog
from mido import MidiFile
import os



input('Please select a music file in midi format:')

root_window = tkinter.Tk()
root_window.withdraw()

def choose_file():
    return filedialog.askopenfilename()

selected_file = choose_file()

print(selected_file)

mid = MidiFile(selected_file)
for i, track in enumerate(mid.tracks):
    print('Track {}: {}'.format(i, track.name))
for msg in track:
    print(msg)



