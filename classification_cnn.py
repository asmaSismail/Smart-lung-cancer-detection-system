
from tensorflow import experimental
from tensorflow.keras import applications
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Convolution2D,Activation,Flatten,Dense,Dropout,MaxPool2D,BatchNormalization
import matplotlib.pyplot as plt
import numpy as np 


# model cnn de classification élu pour la classification
def create_model():
    SIZE=128
    INPUT_SHAPE = (SIZE, SIZE, 1) 
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, (3, 3), kernel_initializer = 'he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), kernel_initializer = 'he_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    return model

#mythreshold=0.5
mythreshold=0.570414