import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

# Criar um modelo Sequential
model = Sequential()

# Adicionar a primeira camada de convolução com dois filtros
model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(5, 5, 1), padding='same'))

# Adicionar a segunda camada de convolução com dois filtros
model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same'))

# Sumário do modelo
model.summary()

# Array de exemplo para representar uma imagem de entrada
input_image = np.array([[[[0.1], [0.2], [0.3], [0.4], [0.5]],
                         [[0.6], [0.7], [0.8], [0.9], [0.1]],
                         [[0.2], [0.3], [0.4], [0.5], [0.6]],
                         [[0.7], [0.8], [0.9], [0.1], [0.2]],
                         [[0.3], [0.4], [0.5], [0.6], [0.7]]]])

# Obter a saída da primeira camada de convolução
conv1_output = model.predict(input_image)

# Obter a saída da segunda camada de convolução
conv2_output = model.predict(conv1_output)

def visualize_image():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Saída da Imagem', fontsize=16)
    for i in range(input_image.shape[-1]):
        axes[i].imshow(input_image[0, :, :, i])
        axes[i].axis('off')
        for j in range(input_image.shape[1]):
            for k in range(input_image.shape[2]):
                axes[i].text(k, j, f'{input_image[0, j, k, i]:.2f}', color='white', ha='center', va='center')
                
# Função para visualizar as saídas das camadas de convolução
def visualize_conv_output(conv_output, layer_num):
    fig, axes = plt.subplots(1, conv_output.shape[-1], figsize=(10, 4))
    fig.suptitle(f'Saída da Camada de Convolução {layer_num}', fontsize=16)
    for i in range(conv_output.shape[-1]):
        axes[i].imshow(conv_output[0, :, :, i])
        axes[i].axis('off')
        for j in range(conv_output.shape[1]):
            for k in range(conv_output.shape[2]):
                axes[i].text(k, j, f'{conv_output[0, j, k, i]:.2f}', color='white', ha='center', va='center')
    
visualize_image()

# Visualizar a saída da primeira camada de convolução
visualize_conv_output(conv1_output, 1)

# Visualizar a saída da segunda camada de convolução
visualize_conv_output(conv2_output, 2)




plt.show()