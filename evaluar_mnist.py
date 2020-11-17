import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical  #para pasara a one-hot
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import random
import copy

from matriz_confusion import graficar_matriz_de_confusion

(null, null), (x_test, y_test) = mnist.load_data(path='data')

originalImages = x_test

x_test = x_test.reshape(10000,28,28,1)

x_test = x_test / 255

y_test = to_categorical(y_test, 10)

# CARGAR MODELO: cargar json y crear el modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("Cargado modelo desde disco.")

# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# RECOGIDA DE DATOS, EL PREDICT
Y_pred = loaded_model.predict(x_test)
y_pred = np.argmax(Y_pred, axis=1) # pasamos de one-hot a normal
y_ref = np.argmax(y_test,axis=1)

#El vector TODOS, de 3 dimensiones lo tiene las imagenes test, la pred, y lo real
todos = [(i, predit, np.argmax(real)) for i,(predit, real) in enumerate(zip(y_pred, y_test))]

score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('test loss:', score[0])
print('test accuracy:', score[1])

etiquetas = ['0','1','2','3','4','5','6','7','8','9']
graficar_matriz_de_confusion(y_ref, y_pred, etiquetas)

# Representar numero random
# unidad = todos[random.randint(0,len(todos))]
# plt.title("Predit:"+str(unidad[1])+"; Real:"+str(unidad[2]))
# plt.imshow(originalImages[unidad[0]], cmap='gray')
# plt.show()


