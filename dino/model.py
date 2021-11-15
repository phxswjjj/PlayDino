
from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import SGD, Adam


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same',
              strides=(4, 4), input_shape=(80, 80, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2)) # 2 jump or do nothing
    adam = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=adam)

    return model


if __name__ == '__main__':
    model = create_model()

    print('done')
