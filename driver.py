import UI as ui
import cnnvgg16Implementation as cnn
import numpy as np

toBeAnalyzed = ui.loadUI() #gets the directory for the folder of images to be labeled

predictions = cnn.runModel(toBeAnalyzed)

cnn.makeCSV(predictions)