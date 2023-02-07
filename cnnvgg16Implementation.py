from numpy import loadtxt
import tensorflow
from keras.utils import img_to_array
from keras.models import load_model
import os


def makeCSV():
    file = open('labeledCandids.csv', 'w')
    file.write('File,RelativePath,DateTime,DeleteFlag,CameraNumber,DayNight,Animal,Count\n')
    file.close()
    print('success!')

def decoder(arr):

    print('success!')

CUTOFF = 0.7

def runModel(mainDIR):
    cnn = load_model('vgg16Run.h5')

    for images in os.scandir(mainDIR):
        if  not images.is_file():
            print('oop. directory')
        
        else:
            print('woot! found a file')
            fileType = images.name.split('.')[1]
            if fileType == 'PNG' or fileType == 'JPG' or fileType == 'BMP' or fileType == 'GIF':
                print('found an image!!!')
                
                arr = cnn.predict(img_to_array(os.mainDIR + '/' + images.name))
                print('pErDicTeD')
                print(arr)
        
