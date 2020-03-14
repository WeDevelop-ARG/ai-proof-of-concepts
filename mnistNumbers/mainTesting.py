from keras.datasets import mnist
import numpy as np
from preprocess import preprocess
from buildModel import buildModel
from keras.utils import to_categorical
from keras.preprocessing import image
from preprocess import preprocess
from getHyperparameters import getHyperparameters
from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

def getNumberFromPrediction(prediction):
  return prediction.argmax(axis=1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = preprocess(x_train)
x_test = preprocess(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = buildModel(x_train)
model.summary()

(loss, optimizer, metrics) = getHyperparameters()

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

model.load_weights('model.h5')

# evaluation tests the model on the whole set
# evaluation = model.evaluate(x_test, y_test)
# print(evaluation)

original_test_image = image.load_img(
    './tests/weird8.jpg', color_mode='grayscale', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))

test_image = image.img_to_array(original_test_image)

test_sample = np.reshape(test_image, (1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

y_predicted = model.predict(test_sample)

print(getNumberFromPrediction(y_predicted))
