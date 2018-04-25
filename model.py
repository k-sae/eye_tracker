import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import rmsprop, SGD

from core import DATA_SET_COLS

saved_weights_name = 'CNN_Weights.h5'
images_shape = (96, 96, 1)


def get_model(model: str = 'CNN') -> Sequential:
    global saved_weights_name
    saved_weights_name = model + "_" + "Weights.h5"
    return {
        'CNN': get_cnn,
        'KNN': get_knn,
        'SVM': get_svm,
    }[model]()


def get_cnn() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # !- may kernel size cause crash if it did replace it with (3,3,1)
    # first conv layer
    model.add(Conv2D(32, 5, activation='relu', input_shape=images_shape))
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
    model.compile(optimizer='adam', loss='mse', metrics=["accuracy"]
                  )
    return model


def get_knn() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # TODO: Build KNN model
    model.add(Conv2D(32, 5, activation='relu', input_shape=images_shape))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, 5, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(256, 2, activation='relu'))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(32,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Dense(len(DATA_SET_COLS)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def get_svm() -> Sequential:
    """

    :rtype: Sequential
    """
    model = Sequential()

    # TODO: Build SVM model

    return model
