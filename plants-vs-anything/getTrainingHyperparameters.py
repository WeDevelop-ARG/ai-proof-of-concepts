from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

def getTrainingHyperparameters():
  CALLBACK = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=15, min_lr=1e-5, verbose=1),
    EarlyStopping(patience=10, restore_best_weights=True)
  ]

  return (CALLBACK)