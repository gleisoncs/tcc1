# Applying thresholding (filtro bilateral)
# https://professor.luzerna.ifc.edu.br/ricardo-antonello/wp-content/uploads/sites/8/2017/02/Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-1.pdf

import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
img = image[::2,::2] # Diminui a imagem

# Aplicar 
suave = np.vstack([
 np.hstack([img,
 cv2.GaussianBlur(img, ( 3, 3), 0)]),
 np.hstack([cv2.GaussianBlur(img, ( 5, 5), 0),
 cv2.GaussianBlur(img, ( 7, 7), 0)]),
 np.hstack([cv2.GaussianBlur(img, ( 9, 9), 0),
 cv2.GaussianBlur(img, (11, 11), 0)]),
 ])


# Exibir a imagem
cv2.namedWindow('Binarization Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binarization Image', 1800, 800)
cv2.imshow('Binarization Image', suave)
cv2.waitKey(0)
cv2.destroyAllWindows()