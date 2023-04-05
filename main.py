import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

from resize import image_resize
from square import squarize
from score import get_score

(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# training_images = training_images[:20000]
# training_labels = training_labels[:20000]

# testing_images = testing_images[:4000]
# testing_labels = testing_labels[:4000]

# To load the saved model
model = models.load_model('model')

img = cv.imread('deer.jpg')
img = image_resize(squarize(img), size=(32, 32))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)

index = np.argmax(prediction)
print(get_score(prediction))
print(f'Prediciton is: {class_names[index]}')

plt.show()
