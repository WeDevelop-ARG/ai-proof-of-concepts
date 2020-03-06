from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Model

def buildModel(x_train):
    i = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    d = Conv2D(filters=8, kernel_size=(5, 5), activation='relu')(i)
    d = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(d)
    d = MaxPool2D(pool_size=(2, 2))(d)
    d = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(d)
    d = Conv2D(filters=64, kernel_size=(2, 2), activation='relu')(d)
    d = Flatten()(d)
    d = Dropout(0.5)(d)
    d = Dense(10, activation='softmax')(d)

    model = Model(inputs=i, outputs=d)

    return model
