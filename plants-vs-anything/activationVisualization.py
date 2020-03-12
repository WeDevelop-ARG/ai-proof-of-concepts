#Modelo
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keract import get_activations, display_activations
from buildModel import buildModel
from getHyperparameters import getHyperparameters
from keras.preprocessing import image
from constants import IMAGE_SIZE

model = buildModel()

(LOSS, OPTIMIZER, METRICS) = getHyperparameters()

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

model.load_weights('model.h5')

original_test_image = image.load_img('./tests/cannabis.jpg', target_size=(IMAGE_SIZE,IMAGE_SIZE))
    
test_image = image.img_to_array(original_test_image)

test_tensor = np.reshape(test_image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))

test_tensor = test_tensor / 255

activations = get_activations(model, test_tensor)

[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

display_activations(activations, save=True, directory="./activations/")