import PySimpleGUI as sg
from tkinter import filedialog as fd

def loadUI():
    filename = ""
    layout = [[sg.Text("Please choose the folder with photos to analyze:")], 
            [sg.Button("Browse Files")], 
            [sg.Button("CANCEL")]]

    # Create the window
    window = sg.Window("MAMMALS UI 0.1", layout)

    # Create an event loop
    isSelected = False
    while not isSelected:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button
        if  event == sg.WIN_CLOSED or event == "CANCEL":
            isSelected = True
        elif event == "Browse Files":
            filename = fd.askdirectory()
            isSelected = True
    window.close()

    return filename