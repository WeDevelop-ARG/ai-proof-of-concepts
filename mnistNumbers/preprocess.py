import numpy as np
from constants import IMAGE_CHANNELS

def preprocess(x):
    x_scaled = x / 255.0
    x_shaped = np.reshape(
        x_scaled, (x.shape[0], x.shape[1], x.shape[2], IMAGE_CHANNELS))

    return x_shaped
