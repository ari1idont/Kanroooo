#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.utils import to_categorical,np_utils
from keras import backend as k
import numpy as np
import pandas as pd
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
#loading dataset
#(xTrain,yTrain),(xTest,yTest)=mnist.load_data()
alpha_data=pd.read_csv('alphabets.csv').astype('float32')
alpha_data.rename(columns={'0':'label'}, inplace=True)
x = alpha_data.drop('label',axis = 1)
y = alpha_data.label
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2)
#xtrain.shape, xtest.shape, ytrain.shape, ytest.shape
import matplotlib.pyplot as plt
#for i in range(6):
#    plt.subplot(int('23'+str(i+1)))
#    plt.imshow(xTrain[i],cmap=plt.get_cmap('gray'))
#reshaping
#xTrain=xTrain.reshape(xTrain.shape[0],28,28,1).astype('float32')
#xTest=xTest.reshape(xTest.shape[0],28,28,1).astype('float32')
#conversion to binary class
#yTrain=to_categorical(yTrain)
#yTest=to_categorical(yTest)
#xTrain=xTrain/255x
#xTest=xTest/255
scaler = MinMaxScaler()
scaler.fit(xTrain)
#scaling data 
xTrain = scaler.transform(xTrain)
xTest = scaler.transform(xTest)
xTrain = np.reshape(xTrain, (xTrain.shape[0], 28,28,1)).astype('float32')
xTest = np.reshape(xTest, (xTest.shape[0], 28,28,1)).astype('float32')
yTrain = np_utils.to_categorical(yTrain,num_classes=26,dtype=int)
yTest = np_utils.to_categorical(yTest,num_classes=26,dtype=int)
yTrain.shape,yTest.shape
letters_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',
             7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',
             14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',
             21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
#show 
fig, axis = plt.subplots(3, 3, figsize=(20, 20))
for i, ax in enumerate(axis.flat):
    ax.imshow(xTrain[i].reshape(28,28))
    ax.axis('off')
    ax.set(title = f"Alphabet : {letters_dict[yTrain[i].argmax()]}")
#defining cnn model
def create_model():
    num_classes=26
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
with open("model_alpha.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_alpha.h5")
#model.save('model.h5')
print('saved')
scores=model.evaluate(xTest,yTest,verbose=0)
print("error: %.2f%%"%(100-scores[1]*100))
