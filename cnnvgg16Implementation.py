import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image

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

    imagesToPredict = tf.keras.utils.image_dataset_from_directory(directory = mainDIR,
                                                                  labels = None,
                                                                  color_mode = 'grayscale',
                                                                  batch_size = 1,
                                                                  image_size = (224,224),
                                                                  shuffle = False,
                                                                  validation_split = None,
                                                                  subset = None)
    def prepImages(dataset):
     imgList = [] #list of all RGB images
    
     #for every setOfBatches in dataset:
     #convert img to rgb, add it to imgList
     #add label to imgLabels
     print('=========== PREPROCESSING: ===========\n')
     for image in dataset:
        print('processing image: ...')
        image = tf.image.grayscale_to_rgb(image)
        imgList.append(image)
     print("=========== PREPROCESSING COMPLETE ===========")
    #creates a new BatchDataset from imgList and imgLabels
     newTrainData = tf.data.Dataset.from_tensor_slices((imgList))
     print('new dataset created. tasks complete! \n===========')
     return newTrainData #returns the new dataset
    
    imagesToPredict = prepImages(imagesToPredict)

    print(cnn.predict(imagesToPredict))