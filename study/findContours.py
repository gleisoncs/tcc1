import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leitura de metadados e imagens
images_path = 'roads/'

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
image_path = os.path.join(images_path, "IMG_20240218_093426.jpg")
image = cv2.imread(image_path)
street_image = segment_street(image)

plt.imshow(street_image)
plt.show()