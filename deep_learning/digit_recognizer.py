from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import cv2

# load the mnist dataset
mnist = tf.keras.datasets.mnist

# split the dataset with train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

# define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# fit the train data to the model
model.fit(x_train, y_train, epochs=5)

# evaluate the
model.evaluate(x_test,  y_test, verbose=2)

# p = model.predict(x_test)[0]
#
# np.argmax(p)
# y_test[0]

# read the image
img = cv2.imread('data/nine.jpg', cv2.IMREAD_GRAYSCALE)
# resize the image
img2 = cv2.resize(img, (28, 28))
# save the processed images
cv2.imwrite("data/image_resized.png", img2)

img = img2.reshape(28, 28, -1)
cv2.imwrite("data/image_reshaped.png", img)

img = 1.0 - img/255.0
cv2.imwrite("data/image_inverted.png", img)

# predict the result
p = model.predict(np.asarray(img.transpose(2, 0, 1)))

# get the value of the maximum likely prediction
print('Prediction: {}'.format(np.argmax(p[0])))



