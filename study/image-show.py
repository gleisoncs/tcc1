import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

img = np.array(Image.open('rabbit.jpg'))
plt.figure(figsize=(8,8))
plt.imshow(img)
plt.show()

