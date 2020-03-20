'''
  This module takes the model and a sample and creates images in a folder
  of the activations throughtout the whole net. Seeing the activations is a
  visual tool to check the model behavior, e.g, if some of the latest
  layers do not activate, then the network is probably underfitting.
'''

import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
from keract import get_activations, display_activations
from buildModelFromDisk import buildModelFromDisk
from buildSampleFromDisk import buildSampleFromDisk

model = buildModelFromDisk()
(img, sample) = buildSampleFromDisk('./tests/cannabis.jpg')

activations = get_activations(model, sample)

[print(k, '->', v.shape, '- Numpy array') for (k, v) in activations.items()]

display_activations(activations, save=True, directory="./activations/")