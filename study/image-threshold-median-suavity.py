# Applying thresholding (filtro bilateral)
# https://professor.luzerna.ifc.edu.br/ricardo-antonello/wp-content/uploads/sites/8/2017/02/Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-1.pdf

import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Aplicar Suavização Mediana
img = image[::2,::2] # Diminui a imagem
suave = np.vstack([
 np.hstack([img,
 cv2.medianBlur(img, 3)]),
 np.hstack([cv2.medianBlur(img, 5),
 cv2.medianBlur(img, 7)]),
 np.hstack([cv2.medianBlur(img, 9),
 cv2.medianBlur(img, 11)]),
 ])

# Exibir a imagem
cv2.namedWindow('Median Suavity Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Median Suavity Image', 1800, 800)
cv2.imshow('Median Suavity Image', suave)
cv2.waitKey(0)
cv2.destroyAllWindows()