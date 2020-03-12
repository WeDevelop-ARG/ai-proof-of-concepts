import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from buildModel import buildModel
from getHyperparameters import getHyperparameters
from constants import IMAGE_SIZE

model = buildModel()
model.summary()

(LOSS, OPTIMIZER, METRICS) = getHyperparameters()

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

model.load_weights('./model.h5')

images = [
 './tests/rose.jpg',
 './tests/cannabis.jpg',
 './tests/bulletproofVest.jpg',
 './tests/greenCar.jpg',
 './tests/plant.jpg',
 './tests/yellowCar.jpg',
 './tests/car.jpg',
 './tests/randomPlant.jpg',
 './tests/plane.jpg',
 ] 


for img in images:
    
    original_test_image = image.load_img(img, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    
    test_image = image.img_to_array(original_test_image)
    
    test_sample = np.reshape(test_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
    
    test_sample = test_sample / 255
    
    y_predicted = model.predict(test_sample)
    
    ##plt.imshow(original_test_image)
    
    argmax = y_predicted.argmax(axis=1)
    percentage = y_predicted[0][argmax]

    if (argmax == 0):
        print(str(img) + ' - ' + 'this is not a plant ' + str(percentage))
    else:
        print(str(img) + ' - ' + 'this is a plant ' + str(percentage))
