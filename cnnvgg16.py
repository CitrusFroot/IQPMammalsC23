##### IMPORT SECTION #####
import tensorflow as tf
import keras, os #Provides infrastructure for Neural Network (NN)
from keras.models import Sequential #specifies that our NN is sequential (one layer links to the next, etc.)
from keras.layers import Conv2D, MaxPool2D , Flatten, Dense, RandomCrop
#Conv2D: images are 2D, hence 2D. Tells system to use Convolutional NNs (CNN)
#MaxPool2D: Max Pooling has been found to be the better option for pooling with image identification (try avg at least once jic)
#Flatten: Converts 2D arrays into a single, continuous vector (DO NOT CHANGE!!!)
#Dense: last 3 layers; condenses outputs from previous layers into a smaller output
#RandomCrop allows us to hone our data to focus on specific aspects of an image

import numpy as np
import matplotlib.pyplot as plt #for data visualization

import tensorflow_datasets as tfds
##### SETUP #####

imageResX = 224 #set to camera specifications. best are 64, 256
imageResY = 224 #set to camera specifications. best are 64, 256
batchSize = 2   #set to power of 2 for optimal usage
#Sets the directories as global variables for the sake of convienence
trainDIR = 'E:\All types of images/Training Data/'

#The following sets up the classes we are sorting mammals into
#This is automatically inferred from the program. MAKE SURE ALL SUBDIRECTORIES OF trainDIR are properly labeled!!
#This specifies the order we want them to be organized in. so jackal-front = 0, jackal-side = 1, ... nothing = 7
classNames = ['fox-front', 'fox-side', 'fox-back','jackal-front', 'jackal-side', 'jackal-back', 'other', 'nothing']

##### PREPROCESSING #####
#Creates a layer to randomly crop images in the dataset
#cropLayer = tf.keras.layers.RandomCrop(imageResY, imageResX, seed = 19541912)

#pulls images from dataset, labels them, divides them into testing and training data, and shuffles them in memory
trainData = tf.keras.utils.image_dataset_from_directory(
                                                        directory = trainDIR,
                                                        labels = 'inferred', 
                                                        class_names = classNames, 
                                                        color_mode = 'grayscale', 
                                                        batch_size = batchSize, 
                                                        image_size = (imageResX, imageResY), 
                                                        shuffle = True, 
                                                        validation_split = 0.3, 
                                                        seed = 19121954, 
                                                        subset = 'training')
'''print("==================")
print(trainData) #<BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>
print("\n")
print(trainData.element_spec)
print("\n")
print(trainData.element_spec[0].shape)
trainData.element_spec[0].shape = (None, imageResX, imageResY, 3)
print("\n")
mutateShape = tf.TensorSpec(shape = (None, 224, 224, 3),  #the new correct shape that did not get auto-fixed
                            dtype = trainData.element_spec[0].dtype, #keep the same as before
                            name  = trainData.element_spec[0].name)  #keep the same as before
trainData.element_spec[0] = mutateShape

tempTupleAsList = list(trainData.element_spec) #this converts element_spec (a tuple) into a list, which is mutable
tempTupleAsList[0] = mutateShape               #sets the shape to be mutateShape
new_element_spec = tuple(tempTupleAsList)      #converts the list back into a tuple
trainData.element_spec = new_element_spec


print(trainData.element_spec)


print("\n====================")'''
#The following nested for loop is necessary because 
#for every batch in trainData, and for every image in each batch, do:
#grayscale all the images
#interpret grayscale images as RGB images
print("\n========================")
print(trainData)
print('\n')

batchCount = 1
for batch in trainData:
    i = 0
    print("============================")
    print("batch " + str(batchCount) + ":") #batch size always = 2, (tuple of images, tuple of labels)
    for img in batch[0]:
        print("processing image " + str(i) + ': ...')
        print("shape of img: " + str(img.shape))
        img = tf.image.grayscale_to_rgb(img)
        print("shape of img post processing: " + str(img.shape))
        
        i += 1
    print("batch " + str(batchCount) + " completed.\nBeginning next batch\n================")
    batchCount += 1

trainData = trainData.apply(tf.data.experimental.assert_element_shape(trainData.element_spec[0].shape[0], trainData.element_spec[0].shape[1], trainData.element_spec[0].shape[2], 3))

print("\n========================")
print(trainData)
print('\n')
    
#VGG-16 + ImageNet only works with RGB. the following line of code converts the grayscale color channel into RGB. The image is still gray
##### IMPLEMENTING VGG-16 MODEL #####

#input_shape: image dimensions + color channels
 #include_top: MUST BE FALSE! true would force the size of the architecture. Images that do not comply with size specifications will alter weights!!!
#weights: imagenet for transfer learning
#classes: 8 classes; [jackal, fox] front, side, back, other, nothing. Uncertainty is decided by confidence level
VGG = keras.applications.VGG16(input_shape = (imageResX, imageResY, 3), 
                               include_top = False, 
                               weights = 'imagenet', 
                               classes = len(classNames))

VGG.trainable = False 

model = keras.Sequential([VGG,
                         keras.layers.Flatten(),
                         keras.layers.Dense(units = 256, activation = 'relu'),
                         keras.layers.Dense(units = 256, activation = 'relu'),
                         keras.layers.Dense(units = len(classNames),   activation = 'softmax')])
#we have 3 dense layers (standard CNN framework), the first 2 have 256 units (nodes/neurons), the last has 2
#relu is industry standard; known for being optimal; test with Leaky ReLu for extra performance
#softmax function converts vector of numbers into probability distribution; used to guess what mammal is in image; good for multiclassed datasets (what we are using) + industry standard

#compile the model
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
#optimizer: AdaM performs best in industry. (Experiment with AdaMax. Rising in standard)

##### MODEL SUMMARY SECTION #####
print("\n=========\nMODEL SUMMARY:\n")
model.summary() #prints out a summary table
hist = model.fit(x = trainData, steps_per_epoch = 30, epochs = 5, validation_steps = 5, verbose = 1) #these numbers need to be experimented with 
model.save('vgg16Run.h5')
print('Saved model to disk')

#The following code creates a graph of the accuracy of the modoel
plt.title('VGG-16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['loss'])
plt.legend(['Accuracy', 'Loss'])
#plt.plot(hist.history['val_loss'])
plt.show()