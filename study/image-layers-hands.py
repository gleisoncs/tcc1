import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg', cv2.IMREAD_COLOR)

# Extrair canais de cores
red = image[:,:,0]
green = image[:,:,1]
blue = image[:,:,2]

# Exibir cada canal separadamente
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(red)
plt.title('Red Channel')

plt.subplot(1, 3, 2)
plt.imshow(green)
plt.title('Green Channel')

plt.subplot(1, 3, 3)
plt.imshow(blue)
plt.title('Blue Channel')

plt.tight_layout()
plt.show()