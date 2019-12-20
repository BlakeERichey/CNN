import gym
import corn_env
import os
import numpy as np
from NNEvo import NNEvo


if __name__ == '__main__':
  from keras.utils import to_categorical
  from keras.preprocessing.image import load_img
  from keras.preprocessing.image import img_to_array
  #--------------- Load images ---------------
  def get_images(path='./images/', dims=(256, 256)):
    '''
      dims: (y,x)
    '''
    images = []
    files = [im for im in os.listdir(path) if (im[-3:] in ['jpg', 'png'])]
    for i, filename in enumerate(files):
      if i % 300 == 0:
        print(i)
      image = load_img(path+filename, target_size=dims)
      numpy_image = img_to_array(image)
      images.append(numpy_image)

    return images, len(images)

  paths = ['healthy_corn', 'nl_blight']
  print('Loading images...')
  # paths = ['tests/corn/healthy', 'tests/corn/nlblight']
  dim = 256
  images = []
  classes = []
  for i, p in enumerate(paths):
    images_batch, num_images = get_images(path=f'../images/{p}/', dims=(dim, dim))
    images += images_batch
    classes += [i for _ in range(num_images)] #class = directory number in paths

  #map images and classes to a NN input and output
  images = np.array(images)
  classes = to_categorical(classes)

  print('Image shape', images[0].shape)

  #--------------- Run Environment ---------------
  # env = gym.make('corn_env-v0')

  # print('Observation Space:', env.observation_space)
  # print('Available Actions:', env.action_space.n   )
  # for epoch in range(1):
  #   num_steps = 0
  #   done      = False
  #   envstate  = env.reset(images, classes)
  #   while not done: #perform action/step
  #     action = env.action_space.sample()
  #     observation, reward, done, info = env.step(action)
  #     env.render()
  #     num_steps += 1

  # env.close()

  config = {
    'tour': 2,
    'cores': 1,
    'cxrt': .005,
    'layers': 0, 
    'env': 'corn_env-v0', 
    'elitist': 2,
    'sharpness': 1,
    'cxtype': 'weave',
    'population': 25,
    'mxrt': 'default',
    'transfer': True,
    'generations': 20, 
    'mx_type': 'default',
    'selection': 'tour',
    'fitness_goal': .99, #*
    'random_children': 0,
    'validation_size': None,
    'activation': 'softmax', 
    'nodes_per_layer': [],
  }

  agents = NNEvo(**config)

  for env in agents.envs:
    env.reset(images, classes)

  agents.train(target='./results/cornGA.h5', callbacks=[])
  # agents.evaluate(filename='./results/cornGA_80.0.h5')
  agents.show_plot()
  agents.evaluate()