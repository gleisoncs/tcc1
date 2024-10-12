# Applying thresholding (limiarização)
import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
bin1 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)

resultado = np.vstack([np.hstack([img, suave]), np.hstack([bin1, bin2])]) 

# Exibir a imagem limiarizada (opcional)
cv2.namedWindow('Blur Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blur Image', 1800, 800)
cv2.imshow('Blur Image', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()