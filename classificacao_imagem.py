import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import GlobalAveragePooling2D, Lambda, MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.datasets import cifar10, mnist
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)

X_test = X_test.astype('float32')
X_train = X_train.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


def my_model():
    model = Sequential()

    model.add(Conv2D(32, (2, 2), activation='relu', padding='same', input_shape=(32, 32, 3))) # adiciona convolução com input-shape
    model.add(Conv2D(64, (2, 2), activation='relu')) # adiciona convolução sequencial
    model.add(MaxPooling2D(pool_size=(2, 2))) # adiciona janela de maxpooling
    model.add(Dropout(0.5)) #desativa alguns neuronios

    model.add(Flatten()) ## transforma vetor em array

    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax')) ## retorna vetor de probabilidade

    return model


model = my_model()

checkpoint = ModelCheckpoint('modeloTOP.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='auto') ##criação para alvar o modelo

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1.0e-4), metrics=['accuracy'])

model_details = model.fit(X_train, Y_train, batch_size=32, #grande para diversificar o modelo
                          epochs=2, # numero de interações, vezes que passam o modelo para a rede treinar
                          validation_data=(X_test, Y_test),
                          #callbacks=[checkpoint],
                          verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)

print("Test score: ", score[0])
print("Accuracy score: ", score[1])
