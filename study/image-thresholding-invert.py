# Applying thresholding (limiarização)
import cv2

# Carregar a imagem
image = cv2.imread('IMG_20240218_105450.jpg')

# Converter a imagem para tons de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar limiarização
_, thresholded_image = cv2.threshold(gray_image, 240, 255, cv2.THRESH_BINARY_INV)

# Salvar a imagem limiarizada
#cv2.imwrite('gray_image_thresholded.jpg', thresholded_image)

# Exibir a imagem limiarizada (opcional)
cv2.namedWindow('Thresholded Gray Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Thresholded Gray Image', 1800, 800)
cv2.imshow('Thresholded Gray Image', gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()