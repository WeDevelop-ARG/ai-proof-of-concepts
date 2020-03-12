from keras.optimizers import RMSprop, Adadelta, Adam

def getHyperparameters():
  # OPTIMIZER = RMSprop(learning_rate=0.001)
  OPTIMIZER = 'adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['categorical_accuracy']

  return (LOSS, OPTIMIZER, METRICS)