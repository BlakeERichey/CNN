import os, time, cv2
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

def get_row(image_name):
  return df.loc[df['image'] == image_name]

def get_image(filename, dims=(256, 256)):
  '''
    dims: (y,x)
  '''
  image = load_img(filename, target_size=dims)
  numpy_image = img_to_array(image)
  return numpy_image, filename

def render(image_arr):
  plt.imshow(np.uint8(image_arr)) #render image

def save_image(image_arr, dest='./', filename='image.jpg'):
  plt.imsave(dest+filename, np.uint8(image_arr))

#+---------- Crop Images With Annotations -------------------------------------+
# df = pd.read_csv('annotations_handheld.csv')
# print(df)

# path = './images/rcnn/original_nlblight/images_handheld/'
# dest = './images/rcnn/original_nlblight/preprocessed/'
# files = [im for im in os.listdir(path) if (im[-3:] in ['JPG', 'peg'])]
# print(len(files), files[0])

# j=0
# for i, image_name in enumerate(files):
#   img_data = get_row(image_name)
#   print('Image:', i)
#   # print(img_data)
#   if not img_data.empty:
#     x1 = img_data['x1'].values[0]
#     x2 = img_data['x2'].values[0]
#     y1 = img_data['y1'].values[0]
#     y2 = img_data['y2'].values[0]
#     x1, x2 = min(x1, x2), max(x1, x2)
#     y1, y2 = min(y1, y2), max(y1, y2)
#     # print(x1, x2, y1, y2)
#     if (x2 - x1) > 300 and (y2 - y1) > 300:
#       try:
#         image, _ = get_image(path+image_name, None)
#         preprocessed = image[y1:y2, x1:x2]
#         render(preprocessed)
#         save_image(preprocessed, dest, f'img{j}.jpg')
#         j+=1
#       except Exception as e:
#         print('Image failed', e)

#+---------- Resize Cropped Images --------------------------------------------+
path = './images/nl_blight/'
dest = './images/preprocessed/'

files = [im for im in os.listdir(path) if (im[-3:] in ['JPG', 'peg', 'jpg'])]
for i, image_name in enumerate(files):
  print('Image:', i)
  image, _ = get_image(path+image_name)
  save_image(image, dest=dest, filename=f'img{i}.jpg')
