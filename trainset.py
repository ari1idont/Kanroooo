from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import to_categorical
from keras import backend as k
import numpy as np
import pandas as pd
from keras.models import model_from_json
#loading dataset
(xTrain,yTrain),(xTest,yTest)=mnist.load_data()
import matplotlib.pyplot as plt
for i in range(6):
    plt.subplot(int('23'+str(i+1)))
    plt.imshow(xTrain[i],cmap=plt.get_cmap('gray'))
#reshaping
xTrain=xTrain.reshape(xTrain.shape[0],28,28,1).astype('float32')
xTest=xTest.reshape(xTest.shape[0],28,28,1).astype('float32')
#conversion to binary class
yTrain=to_categorical(yTrain)
yTest=to_categorical(yTest)
xTrain=xTrain/255
xTest=xTest/255
#defining cnn model
def create_model():
    num_classes=10
    input_shape = (28, 28,1)
    model=Sequential()
    model.add(Convolution2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(Convolution2D(64,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
#building model
model=create_model()
model.fit(xTrain,yTrain,validation_data=(xTest,yTest),epochs=5,batch_size=100,verbose=1)
print('Model was trained')
#saveing
model_json=model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
#model.save('model.h5')
print('saved')
scores=model.evaluate(xTest,yTest,verbose=0)
print("error: %.2f%%"%(100-scores[1]*100))
