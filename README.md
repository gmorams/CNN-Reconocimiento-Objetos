# CNN-Reconocimiento-Objetos

Este repositorio tiene un archivo PYTHON para generar el modelo de la red neuronal (cnn.py), y luego para evaluar los resultados de esta se usa los archivos "evaluar", en función de si se trabaja con un grupo de datos local (evaluar_local.py) o un grupo de datos de una base de datos, en mi caso MNIST (evaluar_mnist.py). El archivo "matriz_confusion.py" simplemente genera la matriz de confusión correspondiente al modelo.

El set MNIST (Modified National Institute of Standards and Technology) contiene un total de 70.000 imágenes en escala de gris, cada una con un tamaño de 28x28 pixeles y que contiene un dígito (entre 0 y 9) escrito a mano por diferentes personas. Los sets de entrenamiento y validación tienen 60.000 y 10.000 imágenes respectivamente.

La precisión obtenida usando el modelo desarrollado en mi Trabajo de Recerca (redes convolucional) es cercana al 96%.
