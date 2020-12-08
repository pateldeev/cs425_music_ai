from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

gui = Tk(className='musicAi')
gui.geometry("500x500")

def editProjectWindow():
#window
        gui = Tk(className = 'Edit Project')
        gui.geometry("500x500")
        exportProjectButton = Button(gui, text='Export Project', command = nameProjectWindow, width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
        exportProjectButton.pack()

def nameProjectWindow():

    tkWindow = Tk(className = 'Export Project')
    tkWindow.geometry("500x500")

    fileNameLabel = Label(tkWindow, text="File Name").grid(row=0, column=0)
    fileNameVar = StringVar()
    fileNameEntry = Entry(tkWindow, textvariable=fileNameVar).grid(row=0, column=1)

    save_btn = ttk.Button(tkWindow, text = 'Save File!', command = lambda : SaveFile(), width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
    save_btn.grid()


def SaveFile():
   data = [('All tyes(*.*)', '*.*')]
   file = asksaveasfile(filetypes = data, defaultextension = data)



def UploadAction(event=None):
    filename = filedialog.askopenfilename()
    print('Selected:', filename)

def validateLogin(username, password):
	print("username entered :", username.get())
	print("password entered :", password.get())
	return


def inputSpotify(event=None):
#window
    tkWindow = Tk()
    tkWindow.geometry('400x150')
    tkWindow.title('Spotify Login')

#username label and text entry box
    usernameLabel = Label(tkWindow, text="User Name").grid(row=0, column=0)
    username = StringVar()
    usernameEntry = Entry(tkWindow, textvariable=username).grid(row=0, column=1)

#password label and password entry box
    passwordLabel = Label(tkWindow,text="Password").grid(row=1, column=0)
    password = StringVar()
    passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=1, column=1)
    validateLogin = partial(validateLogin, username, password)

#login button
    loginButton = Button(tkWindow, text="Login", command=validateLogin).grid(row=4, column=0)
    loginButton.pack()

def inputPandora(event=None):
    #window
    tkWindow = Tk()
    tkWindow.geometry('400x150')
    tkWindow.title('Pandora Login')

#username label and text entry box
    usernameLabel = Label(tkWindow, text="User Name").grid(row=0, column=0)
    username = StringVar()
    usernameEntry = Entry(tkWindow, textvariable=username).grid(row=0, column=1)

#password label and password entry box
    passwordLabel = Label(tkWindow,text="Password").grid(row=1, column=0)
    password = StringVar()
    passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=1, column=1)
    validateLogin = partial(validateLogin, username, password)

#login button
    loginButton = Button(tkWindow, text="Login", command=validateLogin).grid(row=4, column=0)
    loginButton.pack()

def inputTidal(event=None):
    #window
    tkWindow = Tk()
    tkWindow.geometry('400x150')
    tkWindow.title('Tidal Login')

#username label and text entry box
    usernameLabel = Label(tkWindow, text="User Name").grid(row=0, column=0)
    username = StringVar()
    usernameEntry = Entry(tkWindow, textvariable=username).grid(row=0, column=1)

#password label and password entry box
    passwordLabel = Label(tkWindow,text="Password").grid(row=1, column=0)
    password = StringVar()
    passwordEntry = Entry(tkWindow, textvariable=password, show='*').grid(row=1, column=1)
    validateLogin = partial(validateLogin, username, password)

#login button
    loginButton = Button(tkWindow, text="Login", command=validateLogin).grid(row=4, column=0)
    loginButton.pack()


def linkAccounts(event=None):
    gui = Tk(className = 'Link Accounts')
    gui.geometry("500x500")

    spotifyButton = Button(gui, text='Link Spotify', command = inputSpotify, width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
    pandoraButton = Button(gui, text='Link Pandora', command = inputPandora, width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
    tidalButton = Button(gui, text='Link Tidal', command = inputTidal, width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')

    spotifyButton.pack()
    pandoraButton.pack()
    tidalButton.pack()

def displaySettings(event=None):
    gui = Tk(className = 'Settings')
    gui.geometry("500x500")
    audioSettings = Button(gui, text='Audio Settings', width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
    accountsSettings = Button(gui, text='Connected Accounts', width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
    appearanceSettings = Button(gui, text='Appearance', width=60, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')

    audioSettings.pack()
    accountsSettings.pack()
    appearanceSettings.pack()

def displayHelp(event = None):
    gui = Tk(className = 'Help')
    gui.geometry("500x500")


editProjectButton = Button(gui, text='Edit Project', command = editProjectWindow, width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
importMusicButton = Button(gui, text='Import Music', command = UploadAction, width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
linkAccountButton = Button(gui, text='Link Accounts', command = linkAccounts, width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
settingsButton = Button(gui, text='Settings' , command = displaySettings, width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')
helpButton = Button(gui, text='Help', width=40, height=3, bg='#0052cc', fg='#ffffff', activebackground='#0052cc', activeforeground='#aaffaa')

# add button to gui window
importMusicButton.pack()
linkAccountButton.pack()
settingsButton.pack()
helpButton.pack()
editProjectButton.pack()


gui.mainloop()
