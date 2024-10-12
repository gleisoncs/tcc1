import numpy as np
import cv2
from PIL import Image  

def load_and_preprocess_image(file_path, target_size):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    #img = img / 255.0
    #return img.reshape((*target_size, 1))
    return img

image = load_and_preprocess_image('random/IMG_20240218_105450.jpg', target_size=(1800, 800))

print(image)

img = Image.fromarray(image) 
img.save('IMG_20240218_105450_recreated.png')