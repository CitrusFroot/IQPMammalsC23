##### IMPORT SECTION #####
import tensorflow as tf
import keras, os #Provides infrastructure for Neural Network (NN)
#Conv2D: images are 2D, hence 2D. Tells system to use Convolutional NNs (CNN)
#MaxPool2D: Max Pooling has been found to be the better option for pooling with image identification (try avg at least once jic)
#Flatten: Converts 2D arrays into a single, continuous vector (DO NOT CHANGE!!!)
#Dense: last 3 layers; condenses outputs from previous layers into a smaller output

import numpy as np
import matplotlib.pyplot as plt #for data visualization

##### SETUP #####

imageResX = 224 #set to camera specifications. best are 64, 256
imageResY = 224 #set to camera specifications. best are 64, 256
batchSize = 8   #set to power of 2 for optimal usage
valSplit = 0.3  #percent of data that is saved for testing

#Sets the directories as global variables for the sake of convienence
trainDIR = "E:\All types of images\Training Data"

# the number of subdirectories within the "Training Data" directory
numSubdirectories = len(list(os.walk(trainDIR)))

#The following sets up the classes we are sorting mammals into
#This is automatically inferred from the program. MAKE SURE ALL SUBDIRECTORIES OF trainDIR are properly labeled!!
#This specifies the order we want them to be organized in. so jackal-front = 0, jackal-side = 1, ... nothing = 7

# classNames = ['fox-front', 'fox-side', 'fox-back','jackal-front', 'jackal-side', 'jackal-back', 'other', 'nothing']

############################### PREPROCESSING ###############################

#Creates a layer to randomly crop images in the dataset
#cropLayer = tf.keras.layers.RandomCrop(imageResY, imageResX, seed = 19541912)

#pulls images from dataset, labels them, divides them into testing and training data, and shuffles them in memory
trainData = tf.keras.utils.image_dataset_from_directory(
                                                        directory = trainDIR,
                                                        labels = 'inferred', 
                                                        color_mode = 'grayscale', 
                                                        batch_size = batchSize, 
                                                        image_size = (imageResX, imageResY), 
                                                        shuffle = True, 
                                                        validation_split = valSplit, 
                                                        seed = 19121954, 
                                                        subset = 'both')

#ImageNet only works with RGB.
#The following function converts grayscale images to RGB, and fixes the dataset
def applyFunc(dataset):
     imgList = [] #list of all RGB images
     imgLabels = [] #list of labels assigned to each image
    
    #for every setOfBatches in dataset:
    #convert img to rgb, add it to imgList
    #add label to imgLabels
     batchCount = 1
     print('========\n', len(dataset), 'batches to process. Beginning ...')
     for setOfBatches in dataset:
        for img in setOfBatches[0]: #setOfBatches[0] = images
            img = tf.image.grayscale_to_rgb(img) #converts image to RGB format
            imgList.append(img) #adds to list

        for label in setOfBatches[1]: #setOfBatches[1] = labels
            imgLabels.append(label) #adds to list
        print('batch ', batchCount, 'completed. ', (round((batchCount/len(dataset) * 100), 2)), '%', ' finished.')
        batchCount += 1

    #creates a new BatchDataset from imgList and imgLabels
     newTrainData = tf.data.Dataset.from_tensor_slices((imgList, imgLabels)).batch(batch_size = batchSize)
     print('new dataset created. tasks complete! \n===========')
     return newTrainData #returns the new dataset

#calls applyFunc and updates trainData
trainTData = trainData[0]
trainVData = trainData[1]

trainTData = trainTData.apply(applyFunc)
trainVData = trainVData.apply(applyFunc)

trainData = [trainTData, trainVData]
'''
[<BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>,
 <BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>]  
'''

"""for setOfBatches in trainData: #uncomment this if you need to observe the images
    for img in setOfBatches[0]:
            plt.imshow(img.numpy())
            plt.show()"""

############################### IMPLEMENTING VGG-16 MODEL ###############################
#TODO explain weights, explain include_top
#input_shape: image dimensions + color channels
#include_top: (EXPLAIN WHY BETTER) MUST BE FALSE! true would force the size of the architecture. Images that do not comply with size specifications will alter weights!!!
#weights: 'imagenet' for transfer learning (VERIFY)
#classes: 8 classes; [jackal, fox] front, side, back, other, nothing. Uncertainty is decided by confidence level
VGG = keras.applications.VGG16(input_shape = (imageResX, imageResY, 3), 
                               include_top = False,
                               weights = 'imagenet', 
                               classes = numSubdirectories)

VGG.trainable = False 

model = keras.Sequential([VGG,
                         keras.layers.Flatten(),
                         keras.layers.Dense(units = 256, activation = 'relu'),
                         keras.layers.Dense(units = 256, activation = 'relu'),
                         keras.layers.Dense(units = numSubdirectories,   activation = 'softmax')])
#we have 3 dense layers (standard CNN framework), the first 2 have 256 units (nodes/neurons), the last has 2
#relu is industry standard; known for being optimal; test with Leaky ReLu for extra performance
#softmax function converts vector of numbers into probability distribution; used to guess what mammal is in image; good for multiclassed datasets (what we are using) + industry standard

#compile the model
model.compile(optimizer = 'adam',                       #AdaM performs best in industry. (Experiment with AdaMax. Rising in standard) sgd?
              loss = 'sparse_categorical_crossentropy', #sparse_categorical_crossentropy because [insert reason] + code doesn't work otherwise
              metrics = ['accuracy'])


##### MODEL SUMMARY SECTION #####
print("\n=========\nMODEL SUMMARY:\n")
model.summary() #prints out a summary table

#runs the model and saves it as a History object
es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3) #stops training the network if overfitting occurs
hist = model.fit(x = trainData[0],         #these numbers need to be experimented with 
                 steps_per_epoch = None, 
                 epochs = 15,
                 callbacks = es1,
                 validation_data = trainData[1],
                 validation_steps = None, 
                 verbose = 1)           #should be 2 in final system

model.save('vgg16Run.h5') #saves the model as a readable file
print('Saved model to disk') #confirmation message

#The following code creates a graph of the accuracy of the modoel

plt.title('VGG-16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

plt.title('VGG-16 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Loss', 'Validation Loss'])
plt.show()
