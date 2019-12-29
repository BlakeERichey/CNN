import os, time
import tensorflow.keras as keras
import numpy  as np
import pandas as pd
import tensorflow as tf
from os import listdir
from matplotlib import axes
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix
from keras import layers
from keras.utils import to_categorical
from keras.applications import ResNet50, InceptionV3
from keras.models import Model, load_model
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping

print('Loading Images...')
def get_images(path='./images/', dims=(256, 256)):
  '''
    dims: (y,x)
  '''
  images = []
  files = [im for im in os.listdir(path) if (im[-3:] in ['jpg', 'png'])][:1000]
  for j, filename in enumerate(files):
    print(j)
    image = load_img(path+filename, target_size=dims)
    numpy_image = img_to_array(image)
    images.append(numpy_image)

  return images, len(images)

#get images
paths = ['nl_blight']
images = []
classes = []
dims=(1080,1920)
for i, p in enumerate(paths):
  images_batch, num_images = get_images(path=f'./images/{p}/', dims=dims)
  images += images_batch
  classes += [i for _ in range(num_images)]

#map images and classes to a NN input and output
images = np.array(images)
classes = to_categorical(classes)

import cv2
new_dim = (256, 256)
subimages_per_col = dims[0]//new_dim[0]+1
subimages_per_row = dims[1]//new_dim[1]+1
j=0
print(subimages_per_col, subimages_per_row)
for i, image in enumerate(images):
  print('Image:', i)
  # print(image.shape, type(image), image[:5])
  for subimage in range(subimages_per_col):
    arr = image[max(subimage*new_dim[0], 0):min((subimage+1)*new_dim[0], dims[0])]
    for col in range(subimages_per_row):
      temp = arr[0:new_dim[0], max(col*new_dim[1],0):min(dims[1], (col+1)*new_dim[1])]
      # print(temp.shape)
      if temp.shape == (new_dim[0],new_dim[1],3):
        j+=1
        cv2.imwrite(f'./images/rcnn/img{j}.jpg', temp)