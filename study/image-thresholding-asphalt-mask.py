import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg', cv2.IMREAD_GRAYSCALE)

# Aplicar limiarização para segmentar o asfalto
_, asphalt_mask = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)

# Adicionar tons de cinza mais escuros à máscara do asfalto
darkened_asphalt = cv2.addWeighted(asphalt_mask, 0.5, asphalt_mask, 0.5, 0)

# Exibir a imagem resultante
cv2.namedWindow('Darkened Asphalt Mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Darkened Asphalt Mask', 1800, 800)
cv2.imshow('Darkened Asphalt Mask', darkened_asphalt)

# Esperar por uma tecla pressionada e fechar a janela
cv2.waitKey(0)
cv2.destroyAllWindows()
