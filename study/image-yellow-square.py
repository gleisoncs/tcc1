import cv2
import matplotlib.pyplot as plt

imagem = cv2.imread('random/random2.jpg')
for y in range(0, imagem.shape[0], 10): #percorre linhas
  for x in range(0, imagem.shape[1], 10): #percorre colunas
     imagem[y:y+5, x: x+5] = (255,255,0)

plt.figure(figsize=(10,10))
plt.imshow(imagem, aspect='auto')
#plt.axis("off")
plt.show()