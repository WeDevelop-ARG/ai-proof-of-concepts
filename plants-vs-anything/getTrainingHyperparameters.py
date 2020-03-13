'''
  Hyperparameters that are used only for training phase, e.g, number of epochs,
  batch size.
'''

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def getTrainingHyperparameters():
  CALLBACK = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, verbose=1),
    EarlyStopping(patience=30, restore_best_weights=True),
    ModelCheckpoint('./checkpoint.h5', save_best_only=True)
  ]

  BATCH_SIZE = 64
  EPOCHS = 250

  return (EPOCHS, BATCH_SIZE, CALLBACK)