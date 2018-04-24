import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop, SGD

from core import DATA_SET_COLS

saved_weights_name = 'Weights.h5'
images_shape = (96, 96, 1)


def get_model() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # !- may kernel size cause crash if it did replace it with (3,3,1)
    # first conv layer
    model.add(Conv2D(32, 5, input_shape=images_shape, activation='relu'))
    model.add(MaxPooling2D(2))

    # second conv layer
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(2))

    # third conv layer
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))

    # fourth layer 
    model.add(Conv2D(256, 2, activation='relu'))
    model.add(MaxPooling2D(2))

    # flatten the layers
    model.add(Flatten())

    # add fully connected layers
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(len(DATA_SET_COLS)))

    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"]
                  )
    return model
