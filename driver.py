import UI as ui
import cnnvgg16Implementation as cnn
import jsonReading as jr

CUTOFF = 0.3

choiceTuple = ui.loadUI() #gets a tuple of both the directory of interest, and a value (None, 0, 1) determining what to do
if(choiceTuple[1] != ""):
    match choiceTuple[0]:
        case 0:
            predictions = cnn.runModel(choiceTuple[1])
            count = jr.getCount(choiceTuple[1])
            cnn.makeCSV(choiceTuple[1], predictions, count, CUTOFF)
        case 1:
            print("currently not enabled. Sorry!")
            for i in range(700):
                beans = True

            #cnn.retrain(choiceTuple[1])
        case _:
            print('no choice made- everything is all good. closing application!')