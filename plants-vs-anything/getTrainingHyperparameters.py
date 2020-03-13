'''
  Hyperparameters that are used only for training phase, e.g, number of epochs,
  batch size.
'''

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def getTrainingHyperparameters():
  CALLBACK = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-5, verbose=1),
    EarlyStopping(patience=30, restore_best_weights=True)
  ]

  BATCH_SIZE = 96
  EPOCHS = 250

  return (EPOCHS, BATCH_SIZE, CALLBACK)