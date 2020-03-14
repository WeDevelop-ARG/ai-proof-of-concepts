import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from preprocess import preprocess
from buildModel import buildModel
from getHyperparameters import getHyperparameters
from getTrainingHyperparameters import getTrainingHyperparameters

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = preprocess(x_train)
x_test = preprocess(x_test)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# to_categorical transforms a number into oneHotEncoding. E.g 1 becomes [0,1,0,0,0,0,0,0,0,0]

model = buildModel(x_train)

model.summary()

(loss, optimizer, metrics) = getHyperparameters()
(batch_size, epochs) = getTrainingHyperparameters()

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

history = model.fit(
    x=x_train,
    y=y_train,
    verbose=1,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test),
)

model.save('./model.h5')

# plots model improvements on accuracy over time

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# plots hit and misses on predicting the whole test set

cm = confusion_matrix(y_test.argmax(
    axis=1), model.predict(x_test).argmax(axis=1))

print(cm)
