import matplotlib.pyplot as plt
import numpy as np 
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from buildModel import buildModel
from getHyperparameters import getHyperparameters
from getTrainingHyperparameters import getTrainingHyperparameters
from setSeed import setSeed
from constants import IMAGE_SIZE, CHANNELS, TRAINING_DIR, SEED

# Create data generators to load images from disk and do data augmentation

# setSeed()

(LOSS, OPTIMIZER, METRICS) = getHyperparameters()
(EPOCHS, BATCH_SIZE, CALLBACKS) = getTrainingHyperparameters()

data_generator = ImageDataGenerator(
  rescale=1./255,
  validation_split=0.33
)

train_generator = data_generator.flow_from_directory(
  TRAINING_DIR,
  target_size=(IMAGE_SIZE, IMAGE_SIZE),
  shuffle=True,
  seed=SEED,  
  class_mode='categorical',
  batch_size=BATCH_SIZE,
  subset="training"
)

validation_generator = data_generator.flow_from_directory(
  TRAINING_DIR,
  target_size=(IMAGE_SIZE, IMAGE_SIZE),
  shuffle=True,
  seed=SEED,
  class_mode='categorical',
  batch_size=BATCH_SIZE,
  subset="validation"
)

# Build model

model = buildModel()

model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

model.summary()

# Train

history = model.fit_generator(
  train_generator,
  epochs=EPOCHS,
  validation_data=validation_generator,
  shuffle=True,
  callbacks=CALLBACKS
)

model.save('./model.h5')

# Plot step by step movements of the accuracy metric

plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
