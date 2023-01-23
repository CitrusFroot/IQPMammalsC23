from numpy import loadtxt
import tensorflow
from keras.models import load_model

def runModel(mainDIR):
    cnn = load_model('vgg16Run.h5')
    prediction = cnn.predict(mainDIR)
    print(prediction)

        
        
