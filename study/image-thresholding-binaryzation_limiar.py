# Applying thresholding (limiarização)
import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização
suave = cv2.GaussianBlur(img, (7, 7), 0) # aplica blur
(T, bin) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY)
(T, binI) = cv2.threshold(suave, 160, 255, cv2.THRESH_BINARY_INV)

resultado = np.vstack([np.hstack([suave, bin]), np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])])

# Exibir a imagem binarizada (opcional)
cv2.namedWindow('Binarization Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binarization Image', 1800, 800)
cv2.imshow('Binarization Image', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()