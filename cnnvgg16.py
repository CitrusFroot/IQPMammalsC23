##### IMPORT SECTION #####
import tensorflow as tf
import keras, os #Provides infrastructure for Neural Network (NN)

#The following are used from Keras. They are mentioned here for the programmer's understanding:
    #Conv2D: images are 2D, hence 2D. Tells system to use Convolutional NNs (CNN)
    #MaxPool2D: Max Pooling has been found to be the better option for pooling with image identification
    #Flatten: Converts 2D arrays into a single, continuous vector
    #Dense: last 3 layers; must be dense so that a valid output can be generated

import numpy as np #here for programmer's sake. Some debugging code that is currently commented out requires numpy. Also, some of the code runs numpy behind the scenes
import matplotlib.pyplot as plt #for data visualization

##### SETUP #####

imageResX = 224 #Must be 224 to use transfer learning from ImageNet
imageResY = 224 #Must be 224 to use transfer learning from ImageNet
batchSize = 8   #set to power of 2 for optimal usage. The higher the better.
valSplit = 0.3  #percent of data that is saved for testing


#Sets the directories as global variables for the sake of convienence
trainDIR = "E:\All types of images\Training Data/"

# the number of subdirectories within the "Training Data" directory
numSubdirectories = len(list(os.walk(trainDIR)))

############################### PREPROCESSING ###############################

#pulls images from directory, labels them, shuffles them, and divides them into testing and training data
trainData = tf.keras.utils.image_dataset_from_directory(
                                                        directory = trainDIR,
                                                        labels = 'inferred',                    #The labels are derived from the folder names that have the images
                                                        color_mode = 'grayscale',               #Importing the images in as grayscale standardizes the mix of grayscale and RGB images
                                                        batch_size = batchSize,                 #sets the batch size to batchSize
                                                        image_size = (imageResX, imageResY),    #resizes the images accordingly
                                                        shuffle = True,                         #default is true; here for programmer's sake
                                                        validation_split = valSplit,            #sets the validation split to valSplit
                                                        seed = 19121954,                        #seed chosen arbitrarily; birth and death of Alan Turing (my hero)
                                                        subset = 'both')                        #makes trainData a list, where trainData = [training data dataset object, validation data dataset object]

#ImageNet only works with RGB.
#The following function converts grayscale images to RGB, and fixes the dataset
#dataset: a tf.data.dataset object, in this case, a dataset of images
#returns: a new tf.data.dataset object, in this case, the newly reformatted dataset
def applyFunc(dataset):
     imgList = [] #list of all RGB images
     imgLabels = [] #list of labels assigned to each image
    

     #for every setOfBatches in dataset:
     #convert img to rgb, add it to imgList
     #add label to imgLabels
     print('=========== PREPROCESSING: ===========\n')
     batchCount = 1 #for debugging purposes
     print('========\n', len(dataset), 'batches to process. Beginning ...')
     for setOfBatches in dataset:  #set of batches contains a batch of images and a batch of labels for each image
        for img in setOfBatches[0]: #setOfBatches[0] = batch of images
            img = tf.image.grayscale_to_rgb(img) #converts image to RGB format
            imgList.append(img) #adds to list

        for label in setOfBatches[1]: #setOfBatches[1] = labels
            imgLabels.append(label) #adds to list

        print('batch ', batchCount, 'completed. ', (round((batchCount/len(dataset) * 100), 2)), '%', ' finished.') #more debug text
        batchCount += 1 #even more debug stuff
     print("=========== PREPROCESSING COMPLETE ===========")

    #creates a new BatchDataset from imgList and imgLabels
     newTrainData = tf.data.Dataset.from_tensor_slices((imgList, imgLabels)).batch(batch_size = batchSize)
     print('new dataset created. tasks complete! \n===========')
     return newTrainData #returns the new dataset

#for some reason, you cannot run applyFunc on trainData[0], trainData[1]
#You have to store them separately first, and then recombine them
trainTData = trainData[0]
trainVData = trainData[1]
#Applies the applyFunc to each dataset
trainTData = trainTData.apply(applyFunc)
trainVData = trainVData.apply(applyFunc)

trainData = [trainTData, trainVData] #recombines the two datasets
'''
The following commmented lines show the structure of trainData. Leave this here for debug purposes. It can be useful if an issue arises here
[
 <BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>,
 <BatchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None))>
]  
'''

"""
#uncomment this if you need to observe the images
for setOfBatches in trainData:
    for img in setOfBatches[0]:
            plt.imshow(img)
            plt.show()
"""

############################### IMPLEMENTING VGG-16 MODEL ###############################
#input_shape: (image width, image height, # color channels)
#include_top: MUST BE FALSE! include_top must be excluded to ensure feature extraction occurs. It means that the last 3 layers of the transfered model do not get transfered. This is what we train!!! (and we need to resize the last layer)
#weights: 'imagenet' for transfer learning. It takes the weights from a VGG-16 model trained on ImageNet
#classes: must be set to the number of possible classifications. This shapes the final layer of the model
#8 classes; [jackal, fox] front, side, back, other, empty. Uncertainty is decided by confidence level
VGG = keras.applications.VGG16(input_shape = (imageResX, imageResY, 3), 
                               include_top = False,
                               weights = 'imagenet', 
                               classes = numSubdirectories)

VGG.trainable = False #we dont want to train the first 13 layers, just the last 3.

#we have 3 dense layers (standard CNN framework)
model = keras.Sequential([VGG,                  
                         keras.layers.Flatten(),                                                    #converts the model into a 1D vector
                         keras.layers.Dense(units = 1024, activation = 'selu'),                     #this is the best performing structure out of what was tested    
                         keras.layers.Dense(units = 1024, activation = 'relu'),                     #this is the best performing structure out of what was tested
                         keras.layers.Dense(units = numSubdirectories,   activation = 'softmax')])  #units here must be the same size as the number of possible classifications; each one corresponds to a classification.
                                                                                                    #activation must be softmax here to convert the outputs to probabilities. SUM(units 1-8) = 1


#compile the model
#optimizer: adagrad (best), rmsprop (mid), adadelta (bad), nadam (mid), ftrl(good), sgd (more stable, horrible values)
#sparse_categorical_crossentropy: because we have a non-binary classification (size 8), and we are encoding the predictions as integers
                                  #if we weren't using integers, predictions would be one-hot encoded. In which case, we need to use categorical-cross_entropy
                                  #both loss functions are the same, with the only difference being how predictions are encoded.
model.compile(optimizer = 'adagrad',                       
              loss = 'sparse_categorical_crossentropy', 
              metrics = ['accuracy'])


##### MODEL SUMMARY SECTION #####
print("\n=========\nMODEL SUMMARY:\n")
model.summary() #prints out a summary table of the model
#################################

#runs the model and saves it as a History object
es1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3) #stops training the network if validation loss doesn't improve after 3 epochs
hist = model.fit(x = trainData[0],                 #the training data
                 steps_per_epoch = None,           #None defaults to the number of batches
                 epochs = 15,                      #performed best in experimentation
                 callbacks = es1,                  #Tells the model to monitor es1; in this case, monitor val_loss with patience of 3
                 validation_data = trainData[1],   #the validation data
                 validation_steps = None,          #None defaults to the number of batches
                 verbose = 1)                      #1 shows progress bar. Helps gauge how much is done

model.save('vgg16Run.h5') #saves the model as a readable file
print('Saved model to disk') #confirmation message

#The following code creates a graph of the accuracy of the model
plt.title('VGG-16 Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['Accuracy', 'Validation Accuracy'])
plt.show()

#The following code creates a graph of the loss of the model
plt.title('VGG-16 Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['Loss', 'Validation loss'])
plt.show()