# Applying thresholding (filtro bilateral)
# https://professor.luzerna.ifc.edu.br/ricardo-antonello/wp-content/uploads/sites/8/2017/02/Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-1.pdf

import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('IMG_20240218_105450.jpg')

# Aplicar limiarização
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
resultado = np.vstack([img, lap]) 

# Exibir a imagem
cv2.namedWindow('Binarization Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binarization Image', 1800, 800)
cv2.imshow('Binarization Image', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()