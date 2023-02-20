import PySimpleGUI as sg
from tkinter import filedialog as fd

WIN_DIMENSIONS = (100,50)
#Home window
def loadUI():
    pathname = ""
    #displays the following features
    layout = [[sg.Text("Please select an option:")], 
            [sg.Button("Generate Predictions")], 
            [sg.Button("Retrain System")]]
    
    # Create the window with a title
    window = sg.Window(title = "MIQPC23 V.3.1: Home", layout = layout, margins = WIN_DIMENSIONS)

    # Create an event loop
    isTraining = None  #None: we clicked [X]; True: we clicked "Retrain System"; False: we clicked "Generate Predictions"
    while True:
        event, values = window.read()
        #if the window is closed, stop the program
        if event == sg.WIN_CLOSED:
            break
        #if we chose generate predictions, load the next window (browse files UI)
        elif event == "Generate Predictions":
            isTraining = 0
            window.close()
            pathname = browseFilesUI()
            break
        #if we chose retrain system, load the next window (retrain AI)
        elif event == "Retrain System":
            isTraining = 1
            window.close()
            pathname = retrainAI()
            break
    window.close() #closes the window
    return(isTraining, pathname)
    

#UI for generating predictions
def browseFilesUI():
    pathname = "" #path to folder
    layout = [[sg.Text("Please choose the folder with photos to analyze:")], 
            [sg.Button("Browse Folders")], 
            [sg.Button("CANCEL")]]

    #Create the window
    window = sg.Window("MIQPC23 V.3.1: Predictions", layout)

    #Create an event loop
    goBack = False     #did we click cancel?
    while True:
        event, values = window.read()
        #if press [X], close the window
        if  event == sg.WIN_CLOSED:
            break
        #if we click cancel, go back to the home window
        elif event == "CANCEL":
            goBack = True
            break
        #if we click "Browse Folders", set pathname to the selected folder    
        elif event == "Browse Folders":
            pathname = fd.askdirectory()
            break
    window.close() #closes the window
    #if goBack is true, go back to loadUI()
    if(goBack):
        return loadUI()
    return pathname

#UI for retraining the AI
def retrainAI():
    pathname = ""
    layout = [[sg.Text("Please select a directory that contains both the images to train on, and the Timelapse generated CSV:")], 
            [sg.Button("Browse Files")], 
            [sg.Button("CANCEL")]]
    
    #Create the window
    window = sg.Window("MIQPC23 V.3.1: Retraining", layout)

    #Create an event loop
    goBack = False
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED:
            break
        elif event == "Generate Predictions":
            isTraining = True
            break
        elif event == "CANCEL":
            goBack = True
            break
    window.close()
    if(goBack):
        return loadUI()
    return pathname