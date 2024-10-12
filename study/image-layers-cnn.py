import tensorflow as tf
from tensorflow.keras import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carregar a imagem
image_path = 'roads/IMG_20240218_105450.jpg'
image = Image.open(image_path)
image = image.resize((224, 224))  # Redimensionar para o tamanho esperado

# Converter a imagem para matriz NumPy
image_array = np.array(image)
image_array = image_array / 255.0  # Normalizar os valores dos pixels para [0, 1]

# Adicionar uma dimensão para representar o batch (tamanho 1)
image_array = np.expand_dims(image_array, axis=0)

# Carregar um modelo pré-treinado (por exemplo, VGG16)
model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

# Lista de camadas intermediárias que você deseja visualizar
layer_names = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Extrair as saídas das camadas intermediárias
outputs = [model.get_layer(name).output for name in layer_names]
visualization_model = models.Model(inputs=model.input, outputs=outputs)

# Passar a imagem pelo modelo e obter as ativações das camadas intermediárias
activations = visualization_model.predict(image_array)

# Visualizar as ativações das camadas intermediárias
for i, activation in enumerate(activations):
    plt.figure()
    plt.matshow(activation[0, :, :, 0], cmap='viridis')  # Visualizar a primeira ativação do canal 0
    plt.title(layer_names[i])
    plt.show()
