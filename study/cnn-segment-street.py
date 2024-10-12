import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import datasets, layers, models
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Leitura de metadados e imagens
images_path = 'roads/'
metadata_road = pd.read_csv('metadata_road.csv')

# Define uma função para segmentar as ruas nas imagens
def segment_street(image):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplica limiarização
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # Encontra os contornos das regiões brancas (ruas)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Desenha os contornos na imagem original
    street_mask = np.zeros_like(image)
    cv2.drawContours(street_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    # Aplica a máscara na imagem original para extrair apenas as ruas
    street_image = cv2.bitwise_and(image, street_mask)
    return street_image

# Loop sobre as imagens para segmentar as ruas
images = []
labels = []
for idx, row  in metadata_road.iterrows():
    image_path = os.path.join(images_path, row['filename'])
    image = cv2.imread(image_path)
    street_image = segment_street(image)
    if street_image is not None:
        images.append(cv2.resize(street_image, (128, 128)))  # Redimensiona para o tamanho desejado
        labels.append(row['label'])

# Converte para arrays numpy
images = np.array(images)
labels = np.array(labels)

# Divisão do conjunto de dados
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=20)

# Normalização
train_images, test_images = train_images / 255.0, test_images / 255.0

# Definição da arquitetura da CNN
model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(2))

# Compilação do modelo
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Test accuracy:", test_acc)
