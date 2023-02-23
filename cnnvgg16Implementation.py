import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
from PIL.ExifTags import TAGS

cnn = load_model('vgg16Run.h5') #loads the saved model

#this function will create csv file with all the labels
#listOfInfo: a list of tuples. = [(imageName, prediction, probability, mainDIR, relPathImages),...]
#cutoff: a float number. The probability cutoff for whether or not the prediction should be trusted
#returns: nothing
def makeCSV(mainDIR, listOfInfo, count, cutoff):
    file = open(mainDIR + '\\labeledData.csv', 'w') #create/overwrite the csv file and write to it
    file.write('File,RelativePath,Animal,Count,\n') #add the column titles
    file.write(makeCSVHelper(listOfInfo, count, cutoff)) #add the row from makeCSVHelper
    file.close()

############################################################################################################
#this function reads a csv for the name of an image, and the animal present
#mainDIR: the directory that contains images and a csv
#returns: a list of tuples in the format: [(image name, animal present), ...]
def readCSV(mainDIR):
    filesAndDirectories = os.listdir(mainDIR) #gets a list of all the files and folders in mainDIR
    csvFile = "" #blank string that stores the name of the csv file of interest
    #for each file/folder in filesAndDirectories, if it ends in .CSV and is not the labeledData file, set csv to be this file
    for file in filesAndDirectories:
        if(file.endswith(".CSV") and file != "LABELEDDATA.CSV"): #ensures we have the correct file
            csvFile = file
            break

    file = open(csvFile, 'r')  #this opens the csv for reading
    columnTitles = file.readline().split(separator = ',') #gets the column titles
    imageCol = columnTitles.index("FileName") #finds the column no. with the image name
    animalCol = columnTitles.index("Animal") #finds the column no. with the label of the animal

    imageAndLabel = [] #empty list to store tuples

    #for each row in the file, create a tuple with the information we want
    for row in file:
        rowCells = row.split(separator = ',') #turns the row into a parsable list
        tupleInfo = (rowCells[imageCol], rowCells[animalCol]) #creates the tuple with the image name and label
        imageAndLabel.append(tupleInfo) #adds the tuple to the list

    return imageAndLabel #returns the list

#gets the date and time from an image's metadata
#aTuple: a tuple of information; (imageName, prediction, probability, mainDIR, relPathImages)
#returns: a list of size 2 consisting of the date and the time separately
def getDateAndTime(aTuple):
    dirToImage = aTuple[3] + '\\' + aTuple[4] #gets the full directory to the image
    
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
#listOfInfo: a list of tuples. = [(imageName, prediction, probability, mainDIR, relPathImages),...]
#cutoff: a float number. The probability cutoff for whether or not the prediction should be trusted
#returns: a string: a row in a CSV
def makeCSVHelper(listOfInfo, detectionCount, cutoff):
    finalText = '' #the row that is returned
    #for every tuple in listOfInfo:
    #add the name of the image, the relative path of the image, the DateTime of the image, delete flag, camera number, time of day, animal, and the count to finalText
    #Note: order of concatenation must follow:
    #   File,RelativePath,Animal,Count
    countIndex = 0
    for aTuple in listOfInfo:
        finalText = finalText + aTuple[0] + ',' #adds the name of the file and relative path to the string
        finalText = finalText + aTuple[4] + ','   #adds the relative path of the file to the string
        label = "Review" #what is in the image
        if aTuple[2] >= cutoff: #this means that the AI made a prediction with a suitable confidence
            match aTuple[1]:
                case 0:
                    if "empty" in aTuple[4]:   #empty if aTuple is confident. Else, save it for review
                        label = "Empty"
                    else:
                        label = "Review" 
                case 1:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Fox' #fox-back
                case 2:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Fox' #fox-front
                case 3:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Fox' #fox-side
                case 4:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Jackal' #jackal-back
                case 5:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Jackal' #jackal-front
                case 6:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Jackal' #jackal-side
                case 7:
                    if "empty" in aTuple[4]:   #trust megadetector
                        label = "Empty"
                    else:
                        label = 'Review' #other
                case _:
                    label = "Review"
        
        finalText = finalText + label + ','  #adds the prediction to the image
        if(len(detectionCount) > 0):
            finalText = finalText + str(detectionCount[countIndex]) + ",\n" #adds count to entry and ends the entry for the image in the CSV
        else:
            finalText = finalText + ",\n"
        countIndex += 1
    return finalText

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
    imageNames = []
    relPathImages = []
    for pathname, subdirnames, subfilenames in os.walk(mainDIR):
        if(len(subdirnames) == 0):
            for file in subfilenames:
                relPath = os.path.relpath(pathname, mainDIR)

                imageNames.append(file)  #gets the names of all the files in mainDIR for later purposes
                relPathImages.append(relPath) #gets the relative path of all images for csv generation

    #for every prediction in predictions
    #add a tuple of the image name, highest probable prediction, and the probability of that prediction to listOfPredictions
    if(len(predictions) == len(imageNames) and len(predictions) == len(relPathImages)):
        for i in range(len(predictions)):
            #adds the tuple to the list
            listOfPredictions.append((imageNames[i],
                                        np.argmax(predictions[i]), 
                                        np.max(predictions[i]),
                                        mainDIR,
                                        relPathImages[i]))
    return(listOfPredictions)