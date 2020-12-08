#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os.path #Used to ensure the file does not already exist
from os import path

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

