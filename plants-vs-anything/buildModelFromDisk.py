from getHyperparameters import getHyperparameters
from buildModel import buildModel

def buildModelFromDisk():
  model = buildModel()

  (LOSS, OPTIMIZER, METRICS) = getHyperparameters()

  model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)

  model.load_weights('model.h5')

  return model