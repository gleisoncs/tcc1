import numpy as np
import cv2
from PIL import Image  
import matplotlib.pyplot as plt

M = Image.open('random/random7.jpg')
M = np.asarray(M, dtype=np.float32)/255

#M[:,:,0] = 0
#M[:,:,1] = 0

#M[99:200,:,0] = 0
#M[99:200,:,1] = 0

print(M)

plt.figure(figsize=(3,3))
im = plt.imshow(M, aspect='auto')
print(M.shape)
plt.axis("off")
plt.show()