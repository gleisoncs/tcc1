from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem
image_path = 'roads/IMG_20240218_105450.jpg'
image = Image.open(image_path)

# Separar os canais de cores (R, G, B)
red_channel, green_channel, blue_channel = image.split()

# Converter cada canal para uma matriz NumPy
red_array = np.array(red_channel)
green_array = np.array(green_channel)
blue_array = np.array(blue_channel)

# Visualizar cada canal separadamente
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
#plt.imshow(red_array, cmap='gray')
plt.imshow(red_array)
plt.title('Red Channel')

plt.subplot(1, 3, 2)
#plt.imshow(green_array, cmap='gray')
plt.imshow(green_array)
plt.title('Green Channel')

plt.subplot(1, 3, 3)
#plt.imshow(blue_array, cmap='gray')
plt.imshow(blue_array)
plt.title('Blue Channel')

plt.show()
