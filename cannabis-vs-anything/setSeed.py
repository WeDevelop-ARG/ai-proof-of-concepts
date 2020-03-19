'''
  This is to introduce some predictability to the training phase.
'''

import numpy as np
import random 
import tensorflow as tf
import os
from constants import SEED

def setSeed():
  os.environ['PYTHONHASHSEED']=str(SEED)
  np.random.seed(SEED)
  random.seed(SEED)
  tf.random.set_seed(SEED)