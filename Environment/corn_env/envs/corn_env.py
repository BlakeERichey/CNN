import gym
import numpy as np
from   gym   import error, spaces
from matplotlib import pyplot as plt

class Learn_Corn(gym.Env):

  def __init__(self):
    '''
      Initialize environment variables
    '''

    self.action_space      = spaces.Discrete(2)
    self.observation_space = spaces.Box(low=0, high=255, \
      shape=(256,256,3),dtype=np.float32)

    self.num_correct = 0
    self.times_guessed = 0
    self.images = None
    self.classes = None

  def step(self, action):
    '''
      Offers an interface to the environment by performing an action and 
      modifying the environment via the specific action taken

      Returns: envstate, reward, done, info
    '''
    if self.images is not None and self.classes is not None:
      self.guess = action
      if np.argmax(self.classes[self.times_guessed]) == action:
        reward = 1
        self.num_correct += 1
      else:
        reward = 0
      
      if self.times_guessed == len(self.images)-1:
        done = True
        ob = None
      else:
        done = False
        ob = self.images[self.times_guessed+1]

      self.times_guessed += 1
      self.envstate = ob

      return ob, self.num_correct/self.times_guessed, done, {}


  def reset(self,images=None,classes=None):
    ' Returns environment state after reseting environment variables '
    # print('Attempting to reset', self.images is None)
    if (self.images is None and self.classes is None):
      self.images = images
      self.classes = classes
    
    self.num_correct = 0
    self.times_guessed = 0

    self.envstate = self.images[0]
    self.guess = None

    return self.envstate
  
  def render(self,):
    '''
      This method will provide users with visual representation of what is
      occuring inside the environment
    '''
    # plt.imshow(np.uint8(self.envstate)) #render image
    print('Answer:', self.classes[self.times_guessed-1])
    print('Guess:', self.guess)
    print('Num steps:', self.times_guessed)
    print('Accuracy:', self.num_correct/self.times_guessed)
    print('\n\n')
  
  def get_reward(self):
    '''
      Calculate and return reward based on current environment state
    '''
    pass