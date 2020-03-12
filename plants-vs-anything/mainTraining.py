import matplotlib.pyplot as plt
import numpy as np 
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from buildModel import buildModel
from getHyperparameters import getHyperparameters

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.33)

TRAINING_DIR = './v_data_2'
IMAGE_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10
(LOSS, OPTIMIZER, METRICS) = getHyperparameters()

train_generator = data_generator.flow_from_directory(
  TRAINING_DIR,
  target_size=(IMAGE_SIZE, IMAGE_SIZE),
  shuffle=True,
  seed=13,  
  class_mode='categorical',
  batch_size=BATCH_SIZE,
  subset="training"
)

validation_generator = data_generator.flow_from_directory(
  TRAINING_DIR,
  target_size=(IMAGE_SIZE, IMAGE_SIZE),
  shuffle=True,
  seed=13,
  class_mode='categorical',
  batch_size=BATCH_SIZE,
  subset="validation"
)

model = buildModel()
model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
model.summary()

history = model.fit_generator(
  train_generator,
  epochs=EPOCHS,
  validation_data=validation_generator
)

model.save('./model.h5')

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
