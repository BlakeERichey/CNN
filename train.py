#Implement transfer learning to identify disease presence in corn
import os
import keras
import numpy as np
import pandas as pd
from os import listdir
from matplotlib import pyplot as plt
from matplotlib import axes
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.applications import ResNet50
from keras.datasets import cifar10
from keras.models import Model
from keras import layers
from tensorflow.keras.callbacks  import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

dim = 192
epochs = 30
batch_size = 8
#--------------- Load images ---------------
def get_images(path='./images/', dims=(256, 256)):
  '''
    dims: (y,x)
  '''
  images = []
  files = [im for im in os.listdir(path) if (im[-3:] in ['jpg', 'png'])]
  for filename in files:
    image = load_img(path+filename, target_size=dims)
    numpy_image = img_to_array(image)
    images.append(numpy_image)

  return images, len(images)

#get images
paths = ['coke', 'nacho', 'wb']
images = []
classes = []
for i, p in enumerate(paths):
  images_batch, num_images = get_images(path=f'./images/{p}/', dims=(dim, dim))
  images += images_batch
  classes += [i for _ in range(num_images)]

#map images and classes to a NN input and output
images = np.array(images)
classes = to_categorical(classes)

#shuffle and split
images, images_valid, classes, classes_valid = train_test_split(images, classes, test_size=.15)

#--------------- Build NN ---------------
pretrained = ResNet50(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))
#Set Resnet to non trainable
for layer in pretrained.layers:
  layer.trainable = False

#add FCN  
flattened = layers.Flatten()(pretrained.output)
add_layer = layers.Dense(2, activation='relu')(flattened)
add_layer = layers.Dense(64, activation='relu')(add_layer)
add_layer = layers.Dropout(rate=0.2)(add_layer)
add_layer = layers.Dense(32, activation='relu')(add_layer)
output = layers.Dense(classes.shape[1], activation='softmax', name='output')(add_layer)
pretrained = Model(pretrained.inputs, output)
pretrained.summary()

pretrained.compile(RMSprop(lr=5e-3), 'categorical_crossentropy', metrics=['acc'])\

print('Input shape:', pretrained.input_shape)
print('Output shape:', pretrained.output_shape)

#--------------- Train Model ---------------
pretrained.load_weights('./model/model.h5')
ckpt = ModelCheckpoint('./model/best_model.h5', monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True)
es = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=0, mode='auto', baseline=None, restore_best_weights=True)
try:
   pretrained_history = pretrained.fit(images, classes, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(images_valid, classes_valid), callbacks=[])
finally:
   pretrained.save_weights('./model/model.h5') #problem with keras gpu and modelcheckpoint

#--------------- Graph results ---------------
fig, ax = plt.subplots()
ax.plot(pretrained_history.epoch, pretrained_history.history['val_acc'], label='Pretrained')
ax.legend()
ax.set_xlabel('Epoch Number')
ax.set_ylabel('Accuracy')               

#--------------- Test model on new image ---------------
pretrained.load_weights('./model/model.h5')
for image in images[1:1000]:
  print(pretrained.predict(np.expand_dims(image, axis=0)))

score = pretrained.evaluate(images, classes)
print('Score', score)

images = []
for filename in ['test0', 'test1', 'test2']:
  image = load_img(f'./{filename}.jpg', target_size=(dim, dim))
  numpy_image = img_to_array(image)
  images.append(numpy_image)

for image in images:
  print(pretrained.predict(np.expand_dims(image, axis=0)))
# plt.imshow(np.uint8(images_batch[0])) #render an image




#--------------- Save Model as Protocol Buffer file ---------------
from keras import backend as K
import tensorflow as tf
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


# frozen_graph = freeze_session(K.get_session(),
                              # output_names=[out.op.name for out in pretrained.outputs])

# Save to ./model/tf_model.pb
# pretrained.save_weights('./model/model.h5')
# tf.train.write_graph(frozen_graph, "model", "tf_model.pb", as_text=False)