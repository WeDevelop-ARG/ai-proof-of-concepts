def getHyperparameters():
  OPTIMIZER ='adam'
  LOSS = 'categorical_crossentropy'
  METRICS = ['categorical_accuracy']

  return (LOSS, OPTIMIZER, METRICS)