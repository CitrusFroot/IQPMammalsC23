import json #needed to read json files

#reads a json file in the same directory as code and extracts bounding boxes listed in the file
#returns: a list of lists of bounding boxes
def extractBoundingBoxes(): #TODO: get count of detections, record confidence of detections, etc.
    CONFIDENCETHRESHOLD = 0.5
    boundingBoxes = [] #empty starting list; gets returned

    jsonFile = open('image_recognition_file.json') #opens the json file
    data = json.load(jsonFile) #converts the json file into a python readable format (objects -> dicts, etc. View documentation for more info)
    
    #for each image in data (json file)
    #for each detection in each image
    #if the program was confident in its detection (>=0.7), and it is the best confidence yet,
    #set bbox to be that detection's bounding box
    for image in data["images"]:
        bbox = [] #empty list; default value
        bestConf = CONFIDENCETHRESHOLD #used for boolean logic
        for detection in image["detections"]:
            if (detection["conf"] >= bestConf): #if we are confident in the detection, do:
                bbox = detection["bbox"]
                bestConf = detection["conf"]
        boundingBoxes.append(bbox) #add bbox to boundingBoxes, even if bbox = []
    return boundingBoxes