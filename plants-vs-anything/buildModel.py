from keras.layers import Dense, Input, Conv2D, Flatten
from keras.models import Model

def buildModel():
  i = Input(shape=(224,224,3))
  d = Conv2D(filters=16, kernel_size=(5,5), activation='relu')(i)
  d = Conv2D(filters=32, kernel_size=(5,5), activation='relu')(d)
  d = Flatten()(d)
  d = Dense(2, activation='softmax')(d)

  model = Model(inputs=i, outputs=d)

  return model