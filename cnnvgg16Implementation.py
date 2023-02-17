import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
from PIL.ExifTags import TAGS

cnn = load_model('vgg16Run.h5') #loads the saved model

#gets the date and time from an image's metadata
#aTuple: a tuple of information; (imageName, prediction, probability, mainDIR)
#returns: a list of size 2 consisting of the date and the time separately
def getDateAndTime(aTuple):
    dirToImage = aTuple[3] + '/' + aTuple[0] #gets the full directory to the image
    
    image = Image.open(dirToImage) #opens the image for reading
    metadata = image.getexif() #extracts exif metadata TODO: make work for pngs as well
    value = "" #value is a string that contains the DateTime value (date and time concatenated together)

    #for every tagid in metadata:
    #if the tag is DateTime, do:
    #get the value for DateTime
    for tagid in metadata:
        # getting the tag name instead of tag id
        tagname = TAGS.get(tagid, tagid) #extracts the name from the id
        if(tagname == 'DateTime'): #found a match
            value = metadata.get(tagid) #gets value
            
    return value.split(' ') #returns a list: [DATE, TIME]

#takes in a list (listOfInfo) and a cutoff confidence, and returns a row in the CSV file
#listOfInfo: a list of tuples. = [(imageName, prediction, probability, mainDIR),...]
#cutoff: a float number. The probability cutoff for whether or not the prediction should be trusted
#returns: a string: a row in a CSV
def makeCSVHelper(listOfInfo, cutoff):
    finalText = '' #the row that is returned

    #for every tuple in listOfInfo:
    #add the name of the image, the relative path of the image, the DateTime of the image, delete flag, camera number, time of day, animal, and the count to finalText
    #Note: order of concatenation must follow:
    #   File,RelativePath,DateTime,DeleteFlag,CameraNumber,DayNight,Animal,Count\n
    for aTuple in listOfInfo:
        finalText = finalText + aTuple[0] + ',' + './' + aTuple[0] + ',' #adds the name of the file and relative path to the string

        timeStamp = getDateAndTime(aTuple) #gets dateTime value as a list of date and time
        finalText = finalText + timeStamp[0] + timeStamp[1] + ',' #adds DateTime to string
        finalText = finalText + 'False,,,' #filler; adds deleteFlag, cameraNumber, and DayNight to string

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
        
        finalText = finalText + label + ','  #adds the prediction to the image; TODO: add count
        finalText += '\n' #ends the entry for the image in the CSV
    
    return finalText

#this function will create csv file with all the labels
#listOfInfo: a list of tuples. = [(imageName, prediction, probability, mainDIR),...]
#cutoff: a float number. The probability cutoff for whether or not the prediction should be trusted
#returns: nothing
def makeCSV(listOfInfo, cutoff):
    #if the csv already exists, add to it. TODO: consider simply overwritting. Images may be entered twice, or add handling
    if  os.path.isfile('labeledData.csv'):
        file = open('labeledData.csv', 'a') #open the csv file and append to it
        file.write(makeCSVHelper(listOfInfo, cutoff)) #add the row from makeCSVHelper
        file.close() #close file. We're done with it

    else: #file does NOT exist already
        file = open('labeledData.csv', 'w') #create/overwrite the csv file and write to it
        file.write('File,RelativePath,DateTime,DeleteFlag,CameraNumber,DayNight,Animal,Count\n') #add the column titles
        file.write(makeCSVHelper(listOfInfo, cutoff)) #add the row from makeCSVHelper

#preprocesses the images further for the AI; copied and modified from program that compiled the model
#takes in a dataset of images
#returns a new dataset that has been formatted correctly
def prepImages(dataset):
    imgList = [] #list of all RGB images

    #for every image in dataset:
    #convert img to rgb, add it to imgList
    print('=========== PREPROCESSING: ===========\n')
    for image in dataset:
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
    
    for i in range(len(predictions)):
        print(i)
        #following 3 lines of code are for debug/terminal purposes. Please don't delete for programmer's convenience
        print('file name: ' + imageNames[i])
        print('most likely:', np.argmax(predictions[i]), 100*np.max(predictions[i]), '%')
        print('least likely:', np.argmin(predictions[i]), 100*np.min(predictions[i]), '%\n')

        #adds the tuple to the list
        listOfPredictions.append((imageNames[i], 
                                  np.argmax(predictions[i]), 
                                  np.max(predictions[i]),
                                  mainDIR))
    return(listOfPredictions)