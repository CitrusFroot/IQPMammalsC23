##### IMPORT SECTION #####
import keras,os #Provides infrastructure for Neural Network (NN)
from keras.models import Sequential #specifies that our NN is sequential (one layer links to the next, etc.)
from keras.layers import Conv2D, MaxPool2D , Flatten, Dense
#Conv2D: images are 2D, hence 2D. Tells system to use Convolutional NNs (CNN)
#MaxPool2D: Max Pooling has been found to be the better option for pooling with image identification (try avg at least once jic)
#Flatten: Converts 2D arrays into a single, continuous vector (DO NOT CHANGE!!!)
#Dense: last 3 layers; condenses outputs from previous layers into a smaller output

from keras.preprocessing.image import ImageDataGenerator
#Helps scale images and orient them correctly

import numpy as np

##### SETUP #####
imageResX = 224 #set to camera specifications. best are 64, 256
imageResY = 224 #set to camera specifications. best are 64, 256
channelCount = 3 #color channels. Consider changing based on image colors, although VGG-16 might only take in RGB

trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory='C:\Users\Jacob Reiss\Desktop\IQP Code\Training data',target_size=(imageResX,imageResY))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory='C:\Users\Jacob Reiss\Desktop\IQP Code\Test data', target_size=(imageResX,imageResY))

##### IMPLEMENTING VGG-16 MODEL #####

#input_shape: image dimensions + color channels
#include_top: MUST BE FALSE! true would force the size of the architecture. Images that do not comply with size specifications will alter weights!!!
#weights: imagenet because it is industry standard
VGG = keras.applications.VGG16(input_shape = (imageResX, imageResY, 3), include_top = False, weights = 'imagenet')

VGG.trainable = False #Not sure if needed. Test with and without. Pretty sure this should be removed

model = keras.sequential([VGG,
                          keras.layers.Flatten(),
                          keras.layers.Dense(units = 256, activation = 'relu'),
                          keras.layers.Dense(units = 256, activation = 'relu'),
                          keras.layers.Dense(units = 2,   activation = 'softmax')])
#we have 3 dense layers (standard CNN framework), the first 2 have 256 units (nodes/neurons), the last has 2
#relu is industry standard; known for being optimal; test with Leaky ReLu for extra performance
#softmax function converts vector of numbers into probability distribution; used to guess what mammal is in image; good for multiclassed datasets (what we are using) + industry standard

#compile the model
model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ['accuracy'])
#optimizer: AdaM performs best in industry. (Experiment with AdaMax. Rising in standard)

##### MODEL SUMMARY SECTION #####
model.summary() #prints out a summary table
hist = model.fit_generator(steps_per_epoch = 100, generator = traindata, validation_data = testdata, validation_steps = 5, epochs = 5, verbose = 2) #these numbers need to be experimented with
model.save('vggclf.h5')
