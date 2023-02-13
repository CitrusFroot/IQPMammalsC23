import json
def extractBoundingBoxes():
    boundingBoxes = []

    jsonFile = open('image_recognition_file.json')
    data = json.load(jsonFile)
    for image in data["images"]:
        for detections in image["detections"]:
            bboxes = detections[]


    return boundingBoxes