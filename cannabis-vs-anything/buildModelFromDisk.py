from getHyperparameters import getHyperparameters
from buildModel import buildModel

def buildModelFromDisk(path='model.h5'):
  model = buildModel()

  (LOSS, OPTIMIZER, METRICS) = getHyperparameters()

  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

  model.load_weights(path)

  return model