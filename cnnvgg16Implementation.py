import numpy as np
import tensorflow as tf
from keras.models import load_model
import os

cnn = load_model('vgg16Run.h5') #loads the saved model

def makeCSVHelper(listOfInfo, cutoff):
    finalText = ''

    #must adhere to: File,RelativePath,DateTime,DeleteFlag,CameraNumber,DayNight,Animal,Count
    for aTuple in listOfInfo: #listOfInfo = [(imageName, prediction, probability),...]
        finalText = finalText + aTuple[0] + ',' #adds the name of the file to the string

        label = "" #what is in the image
        if aTuple[2] >= cutoff: #this means that the AI made a prediction with a suitable confidence
            match aTuple[1]:
                case 0:
                    label = "" #empty
                case 1:
                    label = 'Fox' #fox-back
                case 2:
                    label = 'Fox' #fox-front
                case 3:
                    label = 'Fox' #fox-side
                case 4:
                    label = 'Jackal' #jackal-back
                case 5:
                    label = 'Jackal' #jackal-front
                case 6:
                    label = 'Jackal' #jackal-side
                case 7:
                    label = 'Review' #other
                case _:
                    label = "ERROR"
        
        finalText = finalText + label + ',' + './' + aTuple[0] + ',' #adds the prediction and the relative path to the image

        

        finalText += '\n' #ends the entry for the image in the CSV
    
    return finalText

#this function will create csv file with all the labels
def makeCSV(listOfInfo, cutoff):
    if not os.path.isfile('labeledData.csv'):
        file = open('labeledData.csv', 'a')
        file.write(makeCSVHelper(listOfInfo, cutoff))
        file.close()
    else:
        file = open('labeledData.csv', 'w')
        file.write('File,RelativePath,DateTime,DeleteFlag,CameraNumber,DayNight,Animal,Count\n')
        file.write(makeCSVHelper(listOfInfo, cutoff))
        print('success!')



#preprocesses the images further for the AI; copied and modified from program that compiled the model
#takes in a dataset of images
#returns a new dataset that has been formatted correctly
def prepImages(dataset):
    imgList = [] #list of all RGB images

    #for every image in dataset:
    #convert img to rgb, add it to imgList
    print('=========== PREPROCESSING: ===========\n')
    for image in dataset:
        print('processing image: ...') #debug message
        image = tf.image.grayscale_to_rgb(image) #converts image from grayscale to rgb. Since images were standardized to grayscale, we now can convert them back to rgb for the AI to work
        imgList.append(image) #adds the modified image to imgList
    print("=========== PREPROCESSING COMPLETE ===========")
    #creates a new dataset from imgList
    newDataset = tf.data.Dataset.from_tensor_slices((imgList))
    print('new dataset created. tasks complete! \n===========')
    return newDataset #returns the new dataset

#this function runs the neural network on a set of images in a directory mainDIR
#returns a list of tuples, each tuple contains: image name, the prediction, the probability of prediction
def runModel(mainDIR):
    listOfPredictions = [] #final list that gets returned

    #creates a dataset of all the images we are making predictions on
    imagesToPredict = tf.keras.utils.image_dataset_from_directory(directory = mainDIR,          #gets the directory
                                                                  labels = None,                #we arent training an AI, so we dont need labels
                                                                  color_mode = 'grayscale',     #for preprocessing sake. Must be imported in as grayscale for AI to work
                                                                  batch_size = 1,               #batch size doesnt matter, however a batch size of None breaks the code
                                                                  image_size = (224,224),       #for preprocessing sake. Must be set to this size for AI to work
                                                                  shuffle = False,              #No need to shuffle data; we aren't training
                                                                  validation_split = None,      #No need for validation split; we aren't training
                                                                  subset = None)                #No need for a subset; we aren't training

    imagesToPredict = prepImages(imagesToPredict) #runs the prepImages function
    predictions = cnn.predict(imagesToPredict) #runs the function that makes predictions on the dataset
    imageNames = os.listdir(mainDIR) #gets the names of all the files in mainDIR for later purposes

    #for every prediction in predictions
    #add a tuple of the image name, highest probable prediction, and the probability of that prediction to listOfPredictions
    for i in range(0,len(predictions) - 1):
        #following 3 lines of code are for debug/terminal purposes. Please don't delete for programmer's convenience
        print('file name: ' + imageNames[i])
        print('most likely:', np.argmax(predictions[i]), 100*np.max(predictions[0]), '%')
        print('least likely:', np.argmin(predictions[i]), 100*np.min(predictions[0]), '%\n')

        #adds the tuple to the list
        listOfPredictions.append((imageNames[i], 
                                  np.argmax(predictions[i], 
                                  np.max(predictions[0]))))
    return(predictions)