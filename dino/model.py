
import os

from keras.callbacks import TensorBoard
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.optimizers import SGD, Adam


def create_model():
    if os.path.exists('model'):
        return load_model('model')

    model = Sequential()
    # 卷積層: 80*80*4 to 20*20*32
    # filter=濾波器數量, kernel_size=濾波器大小
    # strides=移動幅度, padding=填充方式, input_shape=輸入形狀
    # 4+4n=84, n=20
    model.add(Conv2D(32, (8, 8), padding='same',
              strides=(4, 4), input_shape=(80, 80, 4)))
    # 池化層: 20*20*32 to 10*10*32
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # 卷積層: 10*10*32 to 5*5*64
    # 2+2n=12, n=5
    model.add(Conv2D(64, (4, 4), strides=(2, 2),  padding='same'))
    # 池化層: 5*5*64 to 2*2*64
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # 卷積層: 2*2*64 to 2*2*64
    # 1+1n=3, n=2
    model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding='same'))
    # 池化層: 2*2*64 to 1*1*64
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    # 平展: 64
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(3)) # 3 actions: duck, jump, nothing
    adam = Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=adam)

    return model


if __name__ == '__main__':
    model = create_model()

    print('done')
