import numpy as np
from keras.preprocessing import image
from constants import IMAGE_SIZE, CHANNELS

def buildSampleFromDisk(img_path: str):
  original_test_image = image.load_img(img_path, target_size=(IMAGE_SIZE,IMAGE_SIZE))
    
  test_image = image.img_to_array(original_test_image)

  test_tensor = np.reshape(test_image, (1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

  test_tensor = test_tensor / 255

  return (original_test_image, test_tensor)
