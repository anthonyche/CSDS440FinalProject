from keras import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation, Dropout
from numpy.lib.function_base import add_newdoc_ufunc
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from keras.layers.normalization import batch_normalization
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.utils import np_utils
from keras.losses import categorical_crossentropy
import math 
import argparse

#initialize the model
Alexnet = Sequential()
# create an arguement parser
parser = argparse.ArgumentParser(description="Alexnet with differential privacy")
parser.add_argument("para", metavar = "epsilon", type = "str", help="the val of the parameter epsilon")
parser.add_argument("privacy", metavar="privacy_protect", type="str", help="adding noise to data")
args = parser.parse_args()

epsilon = float(args.para)
addnoise = args.privacy

epoch_size = 20
batch_size = 256

# Data processing and add noise to the data
def privacy_protect(data,epsilon):
    norm_list = []
    for i in range(data.shape[0]):
        norm_list.append(np.linalg.norm(data[i]))
    # get the max norm from the norm list
    max_norm = max(norm_list)

    data = data / max_norm
    noise_term = 1/epsilon
    #generate noise
    noise = np.random.normal(0, noise_term, (data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
    data = data + noise
    #add noise
    return data

#data loader
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train.reshape(-1, 227, 227, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

X_train = X_train[:2000]
y_train = y_train[:2000]

if addnoise == "add_noise":
    X_train = privacy_protect(X_train, epsilon)
    print("Noise added")
else:
    X_train = X_train/255
    X_test = X_test/255

#one-hot encoding
classes = 10
y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

# 1st convolutional layer
Alexnet.add(Conv2D(filters=32, input_shape=(28, 28, 1), kernel_size=(3, 3), strides=(1, 1), padding='valid'))
Alexnet.add(Activation('relu'))

# Max pooling
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))

# 2nd convolutional layer
Alexnet.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
Alexnet.add(Activation('relu'))

# Max pooling
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(1,1), padding='valid'))

# 3rd convolutoinal layer
Alexnet.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
Alexnet.add(Activation('relu'))

# 4th convolutional layer
Alexnet.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
Alexnet.add(Activation('relu'))

# Max pooling
Alexnet.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))

#Fully connected layer
Alexnet.add(Flatten())
# 1st fully connected
Alexnet.add(Dense(512))
Alexnet.add(Activation('relu'))

#Dropouts
Alexnet.add(Dropout(0.3))

#Output layer
Alexnet.add(Dense(10))
Alexnet.add(Activation('softmax'))

Alexnet.summary()

#Compile
Alexnet.compile(loss = categorical_crossentropy, optimizer = 'adam', metrics=['accuracy'])

#Train model
Alexnet.fit(X_train,y_train,epochs=epoch_size,batch_size=batch_size)
loss, accuracy = Alexnet.evaluate(X_train,y_train)

print("\n test loss:" + str(loss))
print("\n test accuracy:" + str(accuracy))



    


