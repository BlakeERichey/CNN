# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 03:27:26 2019

@author: Blake

Test model on new images
"""
import keras, os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt

def get_images(path='./images/', dims=(256, 256)):
  '''
    dims: (y,x)
  '''
  images = []
  files = sorted([im for im in os.listdir(path) if (im[-3:] in ['jpg', 'png'])])
  for filename in files:
    image = load_img(path+filename, target_size=dims)
    numpy_image = img_to_array(image)
    images.append(numpy_image)

  return images, len(images)

#--------------- Load Model --------------- 
model_dir = './model/'
model_name = 'corn_model'
model_path = model_dir+model_name+'.h5'

pretrained = load_model(model_path)
pretrained.summary()

#--------------- Load images ---------------
paths = ['tests/corn/healthy', 'tests/corn/nlblight']
dim = 192
images = []
classes = []
for i, p in enumerate(paths):
  images_batch, num_images = get_images(path=f'./images/{p}/', dims=(dim, dim))
  images += images_batch
  classes += [i for _ in range(num_images)]

#map images and classes to a NN input and output
images = np.array(images)
classes = to_categorical(classes)

print('Image shape', images[0].shape)

for image in images:
  print(pretrained.predict(np.expand_dims(image, axis=0)))

plt.imshow(np.uint8(images[0])) #render image