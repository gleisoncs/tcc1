# -*- coding: utf-8 -*-

from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

data = mnist.load_data()
(x_train, y_train), (x_test, y_test) = data

# x_train.shape
plt.imshow(x_train[300] , cmap = 'gray')
plt.title(y_train[300])
plt.show()

# from keras.models import Sequential
from keras.layers import Dense,Flatten, Conv2D, AveragePooling2D
from keras.models import Sequential

model= Sequential()

model.add(Conv2D(filters= 4 ,kernel_size=(5,5),activation='relu',input_shape=(28,28,1)))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Conv2D(filters= 4,kernel_size=(7,7),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=1)

plt.imshow(x_test[100])
plt.title(y_test[100])
plt.show()

test=x_test[76].reshape(-1,28,28,1)

test

model.predict(test)

# Use the model to make predictions
predictions = model.predict(test)

# The 'predictions' variable now contains the predicted probabilities for each class
# You can print the predicted probabilities or get the predicted class
predicted_class = np.argmax(predictions, axis=-1)

print("Predicted Class:", predicted_class)

