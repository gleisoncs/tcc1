import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing import image
import PIL

# Criar um modelo Sequential
model = Sequential()

# Adicionar a primeira camada de convolução com dois filtros
model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'))

# Adicionar a segunda camada de convolução com dois filtros
model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same'))

# Sumário do modelo
model.summary()

# Carregar uma imagem real
img_path = 'random/random1.png'  # Substitua 'exemplo.jpg' pelo caminho da sua imagem
img = image.load_img(img_path, target_size=(28, 28))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

#f = open('cnn-very-basic-3a.txt', 'w')
#f.write(str(img_array))
#f.close()

#f = open('cnn-very-basic-3b.txt', 'w')
#f.write(str(img_array))
#f.close()


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image = cv2.imread('random11.png')

# Convert the image to RGB (matplotlib uses RGB format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Get the dimensions of the image
height, width, _ = image.shape

# Create a figure and axis object with a larger figsize
fig, ax = plt.subplots(figsize=(20, 20))  # Adjust the figsize as needed

# Plot the image
ax.imshow(image_rgb)

# Plot horizontal grid lines for each pixel row
for y in range(height):
    ax.axhline(y - 0.5, color='black', linewidth=0.5)
    
# Plot vertical grid lines for each pixel column
for x in range(width):
    ax.axvline(x - 0.5, color='black', linewidth=0.5)

# Iterate over each pixel in the image
for y in range(height):
    for x in range(width):
        # Get the pixel values at (x, y)
        pixel_values = image[y, x]
        
        # Add text with pixel values at each pixel location
        #ax.text(x, height - y - 1, str(pixel_values), color='red', fontsize=4, ha='center', va='center')  # Adjust fontsize as needed
        
        #ax.text(x, height - y - 1, f"R:{pixel_values[0]}\nG:{pixel_values[1]}\nB:{pixel_values[2]}", color='blue', fontsize=8, ha='center', va='center')  # Adjust fontsize as needed

        # Write the pixel layer number with its own color
        ax.text(x, height - y - 1.1, f"R:{pixel_values[0]}\n", color=(1, 0, 0), fontsize=8, ha='center', va='center')  # Red color for R
        ax.text(x, height - y - 0.9, f"G:{pixel_values[1]}\n", color=(0, 1, 0), fontsize=8, ha='center', va='center')  # Green color for G
        ax.text(x, height - y - 0.7, f"B:{pixel_values[2]}\n", color=(0, 0, 1), fontsize=8, ha='center', va='center')  # Blue color for B

# Show the plot
plt.show()