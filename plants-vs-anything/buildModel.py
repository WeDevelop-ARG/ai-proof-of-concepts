from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Model
from keras.initializers import Constant
from constants import IMAGE_SIZE

def buildModel():
  i = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
  d = Conv2D(filters=8, kernel_size=(3,3), activation='relu', kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(i)
  d = Conv2D(filters=8, kernel_size=(3,3), activation='relu' , kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(d)
  d = MaxPool2D(pool_size=(2,2))(d)
  d = Conv2D(filters=8, kernel_size=(3,3), activation='relu', kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(i)
  d = Conv2D(filters=8, kernel_size=(3,3), activation='relu' , kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(d)
  d = MaxPool2D(pool_size=(2,2))(d)
  d = Flatten()(d)
  d = Dense(16, activation='relu', kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(d)
  d = Dropout(0.25)(d)
  d = Dense(2, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer=Constant(0.1))(d)

  model = Model(inputs=i, outputs=d)

  return model
