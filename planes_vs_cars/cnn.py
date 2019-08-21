from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 224, 224

train_data_dir = 'v_data/train'
validation_data_dir = 'v_data/test'
nb_train_samples = 400
nb_validation_samples = 100
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# - Initializing the CNN
model = Sequential()

""" Arguments:
  filters: Denotes the number of Feature detectors.
  kernel_size: Denotes the shape of the feature detector.
  (3,3) denotes a 3 x 3 matrix
  input _shape: standardises the size of the input image
  activation: Activation function to break the linearity
"""
model.add(Conv2D(filters=32, kernel_size=(2, 2), input_shape=input_shape))
model.add(Activation('relu'))

# - Pooling Layer
""" Arguments:
  pool_size: the shape of the pooling window.
"""
model.add(MaxPooling2D(pool_size=(2, 2)))

# - Adding a second layer of Convolution and Pooling
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# - Flattening Layer
model.add(Flatten())

# - Full-Connection Layer
# - Adding the Hidden layer
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# - Adding the Output Layer
""" Arguments:
  units: Number of nodes in the layer.
  activation : the activation function in each node.
"""
model.add(Dense(units=1))
model.add(Activation('sigmoid'))

# - Compiling the CNN
""" Arguments:
  optimiser: the optimiser used to reduce the cost calculated
  by cross-entropy
  loss: the loss function used to calculate the error
  metrics: the metrics used to represent the efficiency of
  the model
"""
model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# - Generating Image Data
""" Arguments:
  rescale: Rescaling factor. Defaults to None. If None or 0,
  no rescaling is applied, otherwise we multiply the data by
  the value provided
  shear_range: Shear Intensity. Shear angle in a
  counter-clockwise direction in degrees.
  zoom_range: Range for random zooming of the image.
"""
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# This function lets the classifier directly identify the labels
# from the name of the directories the image lies in.
""" Arguments:
  directory: Location of the training_set or test_set
  target_size : The dimensions to which all images found will be resized.
  Same as input size.
  batch_size : Size of the batches of data (default: 32).
  class_mode : Determines the type of label arrays that are returned.
  One of “categorical”, “binary”, “sparse”, “input”, or None.
"""
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# - Training and Evaluating the model
""" Arguments
  generator: A generator sequence used to train the neural
  network(Training_set).
  samples_per_epoch: Total number of steps (batches of samples) to yield
  from generator before declaring one epoch finished
  and starting the next epoch.
  It should typically be equal to the number of
  samples of your dataset divided by the batch size.
  nb_epoch: Total number of epochs. One complete cycle of predictions of
  a neural network is called an epoch.
  validation_data: A generator sequence used to test and evaluate
  the predictions of the neural network(Test_set).
  nb_val_samples: Total number of steps (batches of samples) to yield
  from validation_data generator before stopping at the
     end of every epoch.
"""
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size
)

"""
The above function trains the neural network using the training set and
evaluates its performance on the test set. The functions returns two metrics
for each epoch ‘acc’ and ‘val_acc’ which are the accuracy of predictions
obtained in the training set and accuracy attained in
the test set respectively.
"""
# - Save our Keras models to file
model.save_weights('model_saved.h5')
