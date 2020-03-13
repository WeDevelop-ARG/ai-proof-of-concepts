import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from buildModel import buildModel
from getHyperparameters import getHyperparameters
from buildModelFromDisk import buildModelFromDisk
from buildSampleFromDisk import buildSampleFromDisk
from constants import IMAGE_SIZE, CHANNELS

def printLabelPredictionForImages(imagesPath: list):
    for img in images:
        (original_test_image, test_sample) = buildSampleFromDisk(img)
        
        y_predicted = model.predict(test_sample)
        
        argmax = y_predicted.argmax(axis=1)
        percentage = y_predicted[0][argmax]

        if (argmax == 0):
            print(str(img) + ' - ' + 'this is not a plant - accuracy on prediction: ' + str(percentage))
        else:
            print(str(img) + ' - ' + 'this is a plant - accuracy on prediction: ' + str(percentage))

model = buildModelFromDisk()

# Test prediction on images. Take into consideraton these are specific samples
# to test the boundaries of the classifier. A test set would be more
# suitable for an evaluation.

images = [
 './tests/cannabis.jpg',
 './tests/plant.jpg',
 './tests/randomPlant.jpg',
 './tests/rose.jpg',
 './tests/bulletproofVest.jpg',
 './tests/greenCar.jpg',
 './tests/yellowCar.jpg',
 './tests/car.jpg',
 './tests/plane.jpg',
 ] 

printLabelPredictionForImages(images)