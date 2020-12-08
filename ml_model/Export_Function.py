#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os.path #Used to ensure the file does not already exist
from os import path
from mingus.core import notes
import mingus.extra.lilypond as pond
from mingus.containers.bar import Bar
#import ly.musicxml as xml
from music21 import *
import music21
import wave
from pydub import AudioSegment

# locations for each file in the test environment along with references to the song locations
# https://musescore.com/openscore/scores/5421848 sample song
# https://musescore.com/openscore/bach-sechs-kleine-pr-ludien-bwv-933-938
pianoTestLoc = "/content/PianoTest2.mxl"
mp3TestLoc = "/content/MP3Test2.mp3"
wavTestLoc = "/content/WavTest2.wav"

# converts mp3 to wav
song = AudioSegment.from_mp3(mp3TestLoc)
song.export("WavTest2.wav", format="wav")

mp3Test = wave.open(wavTestLoc,"r")

s1 = converter.parse(pianoTestLoc)
#s1 = corpus.parse('bach/bwv66.6')

# array for storing song notes
noteList = []

# if the note lasts for some amount of time, it is added to the note list. Empty notes are ignored
for note in s1.flat.elements:
  if note.quarterLength != 0:
    noteList += [note]

print(noteList)

# increases the wait time for each note by one
for note in noteList:
  # used to check the class type. Useful since note and chord classes have different attributes
  # if isinstance(note, music21.note.Note):

  #comments allow you to print before and after wait time for comparison
  #tempLength = note.quarterLength
  note.quarterLength += 1
  #print(tempLength, note.quarterLength)

# prints note list before and after the modification is applied
print(noteList)

#print(part1.streamLength)
  
# creates a bar and puts a few notes inside
b1 = Bar()
b1 + "A"
b1 + "B"
b1 + "C"
b1 + "D"

# checks if bar is valid, then outputs as png
bar = pond.from_Bar(b1)
barImage = pond.to_png(bar, "track")

while True: #breaks when the file is exported
    print("Input Filename:")
    filename = input() #User inputs what they want the file called
    if (path.exists(filename)):#Checks if the filename already exists
        print("File Already Exists")
        print("Input A Different Filename")
    else:    
        f = open(filename, "x")#Creates the file
        f.write("testing")#Writes to the file
        f.close()
        break

