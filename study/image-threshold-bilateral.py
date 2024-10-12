# Applying thresholding (filtro bilateral)
# https://professor.luzerna.ifc.edu.br/ricardo-antonello/wp-content/uploads/sites/8/2017/02/Livro-Introdu%C3%A7%C3%A3o-a-Vis%C3%A3o-Computacional-com-Python-e-OpenCV-1.pdf

import cv2
import numpy as np

# Carregar a imagem
image = cv2.imread('roads/IMG_20240218_095419.jpg')

# Converter a imagem para tons de cinza
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização
suave = np.vstack(
    [np.hstack([img,
               cv2.bilateralFilter(img, 3, 21, 21)]),
    np.hstack([cv2.bilateralFilter(img, 5, 35, 35),   
               cv2.bilateralFilter(img, 7, 49, 49)]),
    np.hstack([cv2.bilateralFilter(img, 9, 63, 63),
               cv2.bilateralFilter(img, 11, 77, 77)])
    ])

resultado = np.vstack([np.hstack([suave])])

# Exibir a imagem
cv2.namedWindow('Binarization Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Binarization Image', 1800, 800)
cv2.imshow('Binarization Image', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()