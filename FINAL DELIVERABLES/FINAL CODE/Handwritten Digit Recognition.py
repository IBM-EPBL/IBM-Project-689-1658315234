import numpy
import tensorflow
from keras.datasets import mnist 
# stack for layers
from keras.models import Sequential 
 #input,middle and output layers forcnn structure
from keras import layers
#dense and flatten layers
from keras.layers import Dense,Flatten
#convolutional layers
from keras.layers import Conv2D
#library for building neural networks built on tensorflow
from tensorflow import keras
from keras.optimizers import Adam
from keras.utils import np_utils

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)
print(y_train.shape)

x_train[1]

y_train[9]

import matplotlib.pyplot as plt 

plt.imshow(x_train[9]) 

#CNN expected format: (batch,height,width,channel)
x_train=x_train.reshape(60000,28,28,1).astype('float32')
x_test=x_test.reshape(10000,28,28,1).astype('float32')

no_of_classes=10
#converts output to binary format
y_train=np_utils.to_categorical(y_train,no_of_classes) 
y_test=np_utils.to_categorical(y_test,no_of_classes)

y_train[9]

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation="relu"))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(number_of_classes,activation="softmax"))

#Compilation of the model
model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['accuracy'])

#Early Stopping and Callbacks
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val-loss', patience=1)

#Test the Model
model.fit(x_train,y_cat_train,
          epochs=15,
          validation_data=(x_test,y_cat_test),
          callbacks=[early_stop])

metrics = pd.DataFrame(model.history.history)

metrics

metrics.plot()

metrics[['loss','val_loss']].plot()

metrics[['accuracy','val_accuracy']].plot()

model.evaluate(x_test,y_cat_test,verbose=0)

from sklearn.metrics import classification_report,confusion_matrix

predict_x=model.predict(x_test) 
classes_x=np.argmax(predict_x,axis=1)

print(classification_report(y_test,classes_x))

print(confusion_matrix(y_test,classes_x))

import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,classes_x))

my_num = x_test[1]

classes_x

plt.imshow(my_num.reshape(28,28))

from tensorflow.keras.models import load_model

model.save('CNN.h5')
print('Model Saved!')
 
savedModel=load_model('CNN.h5')
savedModel.summary()

