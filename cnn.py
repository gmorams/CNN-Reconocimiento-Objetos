from __future__ import division
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import copy
import random
from tensorflow.keras.utils import to_categorical  #para pasara a one-hot

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD

##Importamos el conjunto de datos de Keras
from tensorflow.keras.datasets import mnist


##Cargamos los datos de entrenamiento y etiquetas correspondientes, y los datos de validacion y etiquetas correspondientes
(x_train, y_train), (null, null) = mnist.load_data(path='data')

#Transformamos las imagenes a vector, añadiendo una dimension (antes estaban en 6000,28,28)
x_train = x_train.reshape(60000,28,28,1)
#x_test = x_test.reshape(10000,28,28,1)

x_train = x_train / 255
#x_test = x_test / 255

#pasar a formato one_hot (de 3 a [0,0,0,1])

y_train = to_categorical(y_train, 10)
#y_test = to_categorical(y_test, 10)

###Creacion de la CNN

modelo = Sequential()

# CONV1 Y MAX-POOLING1
modelo.add(Conv2D(filters=6, kernel_size=(5,5), activation='relu', input_shape=(28,28,1)))
modelo.add(MaxPooling2D(pool_size=(2,2)))

# CONV2 Y MAX-POOLING2
modelo.add(Conv2D(filters=16, kernel_size=(5,5), activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2)))

# Aplanar, FC1, FC2 y salida
modelo.add(Flatten())
modelo.add(Dense(120,activation='relu'))
modelo.add(Dense(84,activation='relu'))
modelo.add(Dense(10,activation='softmax'))

#COMPILACION
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#TRAIN
nepochs = 25
history = modelo.fit(x_train,y_train,batch_size=128,epochs=nepochs, validation_split=0.2)
#modelo.fit(x_test,y_test,batch_size=128,epochs=nepochs, validation_split=0.2)

##### ESTUDIO
acc_plot = plt.figure(1)
plt.plot(history.history['accuracy'])
plt.title('Precisión de la CNN')
plt.ylabel('Precisión')
plt.xlabel('Época')
plt.xticks([x for x in range(nepochs)], [x + 1 for x in range(nepochs)])
plt.legend(['Entrenamiento'], loc='upper left')

loss_plot = plt.figure(2)
plt.plot(history.history['loss'])
plt.title('Error en la CNN')
plt.ylabel('Error')
plt.xlabel('Época')
plt.xticks([x for x in range(nepochs)], [x + 1 for x in range(nepochs)])
plt.legend(['Entrenamiento'], loc='upper left')

acc_plot.show()
#loss_plot.show()

#SCORE
score = modelo.evaluate(x_train, y_train, verbose=0)
#print('test loss:', score[0])
print('test accuracy:', score[1]*100)


###GUARDAR MODELO
# serializar el modelo a JSON
modelo_json = modelo.to_json()
with open("model.json", "w") as json_file:
    json_file.write(modelo_json)
# serializar los pesos a HDF5
modelo.save_weights("model.h5")
print("Modelo Guardado!")
