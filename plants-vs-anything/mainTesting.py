import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from buildModel import buildModel
from getHyperparameters import getHyperparameters

model = buildModel()
model.summary()

(LOSS, OPTIMIZER, METRICS) = getHyperparameters()

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

model.load_weights('model.h5')

original_test_image = image.load_img(
    './tests/bulletproofVest.jpg', target_size=(224,224))

test_image = image.img_to_array(original_test_image)

test_sample = np.reshape(test_image, (1, 224, 224, 3))

y_predicted = model.predict(test_sample)

plt.imshow(original_test_image)

if (y_predicted[0][0] > 0.5):
    print('this is not a plant')
else:
    print('this is a plant')
