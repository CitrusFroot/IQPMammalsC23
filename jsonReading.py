import json #needed to read json files

#gets the count of animals in an image
#mainDIR: a string that represents the directory with the images
#returns: a list of integers representing the count of animal detections in each image
def getCount(mainDIR):
    counts = [] #list that gets returned
    jsonFile = open(mainDIR + '/image_recognition_file.json') #opens the JSON file
    data = json.load(jsonFile) #accesses the data in the json file

    #for every image dictionary in the images dictionary in data
    #increase count if a detection dictionary in the image dictionary has a confidence >= 0.1, and is category 1 (animal)
    for image in data["images"]:
        count = 0 #gets added to counts
        for detection in image["detections"]:
            if(detection["conf"] >= .1 and detection["category"] == 1): #if we are confident in the detection, and the detection is category 1 (animal), do:
                count += 1 #increase the count
        counts.append(count) #add the final count to counts
    return counts