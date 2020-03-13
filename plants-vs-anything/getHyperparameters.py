'''
  Hyperparameters used to compile the model
'''

from keras.optimizers import RMSprop, Adadelta, Adam, SGD

def getHyperparameters():
  OPTIMIZER = 'adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['categorical_accuracy']

  return (LOSS, OPTIMIZER, METRICS)