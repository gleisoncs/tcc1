import numpy as np
import cv2

# Criar uma imagem de 300x300 pixels com três camadas
image = np.zeros((300, 300, 3), dtype=np.uint8)

# Definir cores diferentes em cada canal de cor
image[:, :, 0] = 186  # Canal Azul
image[:, :, 1] = 114  # Canal Verde
image[:, :, 2] = 132  # Canal Vermelho

# Sobrepor as camadas para obter a cor resultante
result = cv2.merge((image[:,:,0], image[:,:,1], image[:,:,2]))

# Exibir a imagem resultante
cv2.imshow('Mistura de Cores', result)
cv2.waitKey(0)
cv2.destroyAllWindows()