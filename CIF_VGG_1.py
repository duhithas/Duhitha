# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 17:16:44 2018

@author: duhitha.s
"""

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# =============================================================================
# # Plot ad hoc CIFAR10 instances
# from keras.datasets import cifar10
# from matplotlib import pyplot
# from scipy.misc import fromimage
# # load data
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
# # create a grid of 3x3 images
# for i in range(0, 9):
# 	pyplot.subplot(330 + 1 + i)
# 	pyplot.imshow(toimage(X_train[i]))
# # show the plot
# pyplot.show()
# =============================================================================

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create the model
# =============================================================================
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.2))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# =============================================================================

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# =============================================================================
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_1 (Conv2D)            (None, 3, 32, 32)         9248      
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 3, 32, 32)         0         
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 3, 32, 32)         9248      
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 1, 16, 32)         0         
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 512)               0         
# _________________________________________________________________
# dense_1 (Dense)              (None, 512)               262656    
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 512)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                5130      
# =================================================================
# Total params: 286,282
# Trainable params: 286,282
# Non-trainable params: 0
# _________________________________________________________________
# =============================================================================

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))






