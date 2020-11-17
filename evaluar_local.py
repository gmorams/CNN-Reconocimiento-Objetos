import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical  #para pasara a one-hot
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import cv2
import copy
#
# DIR='./digitos'
# num_fotos = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
# for x in range (1,num_fotos):
#     img = cv2.imread('di')

image_number = 1
while os.path.isfile('digitos/digito{}.png'.format(image_number)):
    try:
        img = cv2.imread('digitos/digito{}.png'.format(image_number))[:,:,0]
        if(img.shape!=(28,28)):
            dim = (28,28)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        img = np.invert(np.array([img]))
        x_test = img.reshape(1, 28, 28, 1)
        x_test = x_test / 255

        # CARGAR MODELO: cargar json y crear el modelo
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # cargar pesos al nuevo modelo
        loaded_model.load_weights("model.h5")
        print("Cargado modelo desde disco.")
        # Compilar modelo cargado y listo para usar.
        loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        prediction = loaded_model.predict(x_test)
        plt.title("Número predecido: " + str(np.argmax(prediction)))
        print("The number is probably a {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap='gray')
        plt.show()
    except:
        print("Error reading image! Proceeding to the next one...")
    finally:
        image_number += 1





