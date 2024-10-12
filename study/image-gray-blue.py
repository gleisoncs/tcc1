import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('random/IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Definir intervalos de cor para tons de azul
lower_blue = np.array([100, 0, 0], dtype=np.uint8)
upper_blue = np.array([255, 100, 100], dtype=np.uint8)

# Filtrar a imagem original para identificar tons de azul
blue_mask = cv2.inRange(image, lower_blue, upper_blue)

# Juntar as máscaras de tons de cinza e azul
combined_mask = cv2.bitwise_or(gray_image, blue_mask)

# Salvar a imagem com tons de cinza e azul
cv2.imwrite('image-gray-blue-script1.jpg', combined_mask)

# Mostrar a imagem resultante
cv2.imshow('Gray and Blue Image', combined_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
