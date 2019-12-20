#Implement transfer learning to identify disease presence in corn
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

dim = 256
epochs = 200
batch_size = 4
model_dir = './model/'
model_name = 'Corn_InceptionV3'
model_path = model_dir+model_name+'.h5' #save model to

load_images         = False #***
load_existing_model = False
train_model         = True
graph_results       = True
evaluate            = True
save_pb             = True
create_tflite       = True
#--------------- Load images ---------------
if load_images:
  print('Loading Images...')
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
  paths = ['healthy_corn', 'nl_blight']
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
  images_train, images_valid, classes_train, classes_valid = train_test_split(images, classes, test_size=.15)

#--------------- Build NN ---------------
if load_existing_model:
  pretrained = load_model(model_path)
else:
  pretrained = InceptionV3(weights='imagenet', include_top=False, input_shape=(dim, dim, 3))
  #Set Resnet to non trainable
  for layer in pretrained.layers:
    layer.trainable = False

  #add FCN  
  flattened = layers.Flatten()(pretrained.output)
  add_layer = layers.Dense(2, activation='relu')(flattened)
#  add_layer = layers.Dense(64, activation='relu')(add_layer)
#  add_layer = layers.Dropout(rate=0.2)(add_layer)
  add_layer = layers.Dense(32, activation='relu')(add_layer)
  output = layers.Dense(classes.shape[1], activation='softmax', name='output')(add_layer)
  pretrained = Model(pretrained.inputs, output)

  pretrained.compile(Adam(lr=1e-3), 'categorical_crossentropy', metrics=['acc'])

pretrained.summary()
print('Input shape:', pretrained.input_shape)
print('Output shape:', pretrained.output_shape)

#--------------- Train Model ---------------
if train_model:
  # ckpt = ModelCheckpoint(model_dir + 'best_model.h5', monitor='val_loss', verbose=0, save_weights_only=True, save_best_only=True)
  es = EarlyStopping(monitor='val_acc', min_delta=0, patience=25, verbose=0, mode='auto', baseline=0.85, restore_best_weights=True)
  pretrained_history = pretrained.fit(
    images_train,
    classes_train,
    verbose=1, 
    callbacks=[es],
    epochs=epochs,
    batch_size=batch_size, 
    validation_data=(images_valid, classes_valid)
  )
  
  pretrained.save(model_path)

#--------------- Graph results ---------------
if graph_results:
  fig, ax = plt.subplots()
  ax.plot(pretrained_history.epoch, pretrained_history.history['val_acc'], label='Pretrained')
  ax.legend()
  ax.set_xlabel('Epoch Number')
  ax.set_ylabel('Accuracy')               

#--------------- Evaluate Model ---------------
if evaluate:
  score = pretrained.evaluate(images, classes)
  print('Metrics', pretrained.metrics_names)
  print('Score', score)

  predictions = pretrained.predict(images)
  matrix = confusion_matrix(classes.argmax(axis=1), predictions.argmax(axis=1))
  print('Confusion Matrix')
  print(matrix)

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

if save_pb:
  frozen_graph = freeze_session(K.get_session(),
    output_names=[out.op.name for out in pretrained.outputs])

  # Save to .pb
  tf.train.write_graph(frozen_graph, model_dir, model_name+'.pb', as_text=False)

#--------------- Convert to tflite model ---------------
if create_tflite:
  print('Saving tflite...')
  converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path)
  tflite_model = converter.convert()
  open(f'{model_dir}{model_name}.tflite', 'wb').write(tflite_model)