import json
def extractBoundingBoxes(): #TODO: get count of detections, record confidence of detections, etc.
    boundingBoxes = []

    jsonFile = open('image_recognition_file.json')
    data = json.load(jsonFile)
    for image in data["images"]:
        if(len(image["detections"] > 0))
        detection1 = image["detections"][0] #first detection.
        bbox = detection1["bbox"] #bounding boxes in form []
        boundingBoxes.append(bbox)


    return boundingBoxes