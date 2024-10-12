# Applying thresholding (filtro bilateral)
# https://professor.luzerna.ifc.edu.br/ricardo-antonello/wp-content/uploads/sites/8/2017/02/Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-1.pdf

import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar Canny
sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
sobel = cv2.bitwise_or(sobelX, sobelY)
resultado = np.vstack([np.hstack([img, sobelX]),np.hstack([sobelY, sobel])]) 

# Exibir a imagem
cv2.namedWindow('Binarization Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binarization Image', 1800, 800)
cv2.imshow('Binarization Image', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()