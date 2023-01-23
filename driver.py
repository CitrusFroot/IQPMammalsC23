import UI as ui
import cnnvgg16Implementation as cnn

toBeAnalyzed = ui.loadUI() #gets the directory for the folder of images to be labeled

cnn.runModel(toBeAnalyzed)

