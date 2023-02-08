from numpy import loadtxt
import tensorflow as tf
from keras.utils import img_to_array
from keras.models import load_model
import os
from PIL import Image
from torchvision import transforms

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

    for images in os.listdir(mainDIR): #gets list of file names AND directory names
        print(images)
        if(images.endswith('.PNG') or images.endswith('.JPG')):
            image = Image.open(images)
            loadedImg = transforms.toTensor(image)
            loadedImg = tf.image.rgb_to_grayscale(loadedImg)
            loadedImg = tf.image.grayscale_to_rgb(loadedImg)
            loadedImg = tf.image.resize(image = loadedImg, size = (224,224))
            
            arr = cnn.predict(loadedImg)
            print('pErDicTeD')
            print(arr)
        
        else:
            print('invalid image or directory provided.')