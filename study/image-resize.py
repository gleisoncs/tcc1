import cv2
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('random/IMG_20240218_105450.jpg')

print(image.shape)

# Define new dimensions
new_width = 64
new_height = 64

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))

# Display the resized image
plt.imshow(resized_image, cmap='gray')
plt.title('Resized Channel')
plt.show()