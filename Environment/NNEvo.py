'''
Created on Wednesday December 18, 2019

@author: Blake Richey

Implementation of NeuroEvolution Algorithm v2.0:
  Develop a neurel network and implement genetic algorithm that finds optimum
  weights, as an alternative to backpropogation.

  Uses multiprocessing to enhance speed of convergence
'''

import gym, operator, time, math, statistics
import os, datetime, random
import numpy             as np
import tensorflow        as tf
import matplotlib.pyplot as plt
from   tensorflow.keras.optimizers   import Adam
from   collections                   import deque
from   tensorflow.keras              import backend
from   sklearn.model_selection       import train_test_split
from   tensorflow.keras.applications import ResNet50
from   tensorflow.python.client      import device_lib
from   tensorflow.keras.models       import Sequential, Model, clone_model
from   tensorflow.keras.callbacks    import TensorBoard, ModelCheckpoint
from   tensorflow.keras.layers       import Dense, Dropout, Conv2D, MaxPooling2D, \
    Activation, Flatten, BatchNormalization, LSTM

import traceback, logging
from multiprocessing import Pool, Process, Queue, Array

#speed up forward propogation
backend.set_learning_phase(0)

#disable warnings in subprocess
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.disable(logging.WARNING)

def deserialize(genes, shapes, lengths):
  '''
    deserializes gene string into weights
  '''

  weights = []
  for i, val in enumerate(lengths):
    if i == 0:
      begin = 0
    else:
      begin = lengths[i-1]
    weights.append(np.array(genes[begin:val]).reshape(shapes[i]))

  return weights

def multi_quality(
  res=None, 
  env=None,
  layers=1,
  shapes=(1,),
  lengths=(1,), 
  inputs=None,
  outputs=1,
  genes=None,
  index=None,
  sharpness=1,
  activation='linear',
  nodes_per_layer=[128],
  transfer=False):

  '''
    implements multiprocessed nn evaluation on a gym environment
    res: results are indexed into res at `index` 
  '''
  try:
    genes = [val for val in genes]
    print(f'Testing model {index}')
    if not transfer:
      model = Sequential()
      model.add(Dense(inputs, input_shape = (inputs,)))
      
      for layer in range(layers):

          try:
              nodes=nodes_per_layer[layer]
          except IndexError:
              nodes = None

          if nodes is None:
              nodes = 128

          model.add(Dense(units = nodes, activation = 'relu'))
      
      #output layer
      model.add(Dense(units = outputs, activation = activation))
      model.compile(optimizer = Adam(lr=0.001), loss = 'mse', metrics=['accuracy'])
    elif transfer:
      model = ResNet50(weights='imagenet', include_top=False, input_shape=(env.observation_space.shape))
      for layer in model.layers:
        layer.trainable = False
      
      flattened = Flatten()(model.output)
      #Add FCN
      for layer in range(layers):

        try:
            nodes=nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = 128

        if layer == 0:
          add_layer = Dense(units = nodes, activation = 'relu')(flattened)
        else:
          add_layer = Dense(units = nodes, activation = 'relu')(add_layer)
      
      if layers:
        output = Dense(units = outputs, activation = activation)(add_layer)
      else:
        output = Dense(units = outputs, activation = activation)(flattened)

      model = Model(model.inputs, output)
      model.compile(Adam(lr=1e-3), 'mse', metrics=['acc'])

    if transfer:
      fcn_weights = deserialize(genes, shapes, lengths)
      assert len(fcn_weights) == len(shapes), \
        f'Invalid Weight Structure. Expected {len(shapes)}, got {len(fcn_weights)}.'
      all_weights = model.get_weights()
      untrainable = all_weights[:-len(shapes)]
      weights = all_weights[-len(shapes):]
      for i, matrix in enumerate(weights):
        matrix[:] = fcn_weights[i]
    
      model.set_weights(untrainable + weights)
    else:
      weights = deserialize(genes, shapes, lengths)
      model.set_weights(weights)
    
    # print('index', index)
    # print('genes', genes)
    # print('weights', weights, '\n\n\n')

    total_rewards = []
    for epoch in range(sharpness):
      done = False
      rewards = []
      envstate = env.reset()
      while not done:
        #adj envstate
        if transfer:
          envstate = np.expand_dims(envstate, axis=0)
        else:
          envstate = envstate.reshape(1, -1)
        
        qvals = model.predict(envstate)[0]
        if outputs == 1:
          action = qvals  #continuous action space
        else:
          action = np.argmax(qvals) #discrete action space

        envstate, reward, done, info = env.step(action)
        rewards.append(reward)
      
      total_rewards.append(reward)
    
    # if 5 >= sharpness >= 1:
    #   result = max(total_rewards)
    # else:
    result = sum(total_rewards)/len(total_rewards)
  except Exception as e:
    print('Exception Occured in Process!', e)
    result = -1000000
  print(f'Model {index} Results: {result}')
  res[index] = result

  # spontaneous saving
  if result > .79:
    print(f'Saving model {index}...')
    model.save_weights(f'./results/cornGA_{str(round(result, 2)*100)[:-2]}.h5')
    print('Model saved')
  return result


class NNEvo:

  def __init__(self, 
    tour=2,
    cores=1,
    cxrt=.01,
    layers=1, 
    env=None,
    elitist=3,
    sharpness=1, 
    cxtype='weave',
    population=15, 
    mxrt='default', 
    transfer=False,
    generations=50, 
    selection='tour',
    mx_type='default',
    random_children=0, 
    fitness_goal=None,
    validation_size=0,
    activation='linear',
    nodes_per_layer=[128]):

    '''
      config = {
      'tour': 2,
      'cores': 1,
      'cxrt': .005,
      'layers': 2, 
      'env': 'MountainCar-v0', 
      'elitist': 3,
      'sharpness': 1,
      'cxtype': 'weave',
      'population': 30,
      'mxrt': 'default',
      'transfer': False,
      'generations': 200, 
      'mx_type': 'default',
      'selection': 'tour',
      'fitness_goal': -110,
      'random_children': 0,
      'validation_size': 100,
      'activation': 'linear', 
      'nodes_per_layer': [256,256],
    }
    '''

    self.default_nodes   = 128
    self.mxrt            = mxrt        #chance of a single weight being mutated
    self.cxrt            = cxrt        #chance of parent being selected (crossover rate)
    self.best_fit        = None        #(model, fitness) with best fitness
    self.tour            = tour        #tournament sample size when using tour selection policy
    self.cores           = cores       #how many cores to run forward propogation on
    self.cxtype          = cxtype      #cross over type (gene splicing or avging)
    self.goal_met        = False       #holds model number that meets fitness goal
    self.num_layers      = layers      #qty of hidden layers
    self.mx_type         = mx_type
    self.elitist         = elitist     #n best models transitioned into nxt gen
    self.transfer        = transfer    #implement transfer cnn
    self.sharpness       = sharpness   #epochs to run when evaluating fitness
    self.selection_type  = selection   #selection type (cxrt/tour)
    self.activation      = activation  #activation type for output layer
    self.pop_size        = population  #number of neural nets in population
    self.generations     = generations 
    self.fitness_goal    = fitness_goal #goal for fitness (episode score) to reach
    self.validation_size = validation_size #number of episodes to run to validate a models success in reaching a fitness goal
    self.nodes_per_layer = nodes_per_layer #list of qty of nodes in each hidden layer
    self.random_children = random_children #how many children to randomly mutate
    
    #create environments
    self.envs = [gym.make(env) for _ in range(self.cores)]
    print('Environments Created:', self.cores)
    self.num_features = self.envs[0].observation_space.shape[0]

    
    outputs = 1
    if hasattr(self.envs[0].action_space, 'n'):
      outputs = self.envs[0].action_space.n
    self.num_outputs     = outputs

    self.models = [] #list of individuals 
    self.pop    = [] #population (2d-list of weights)
    self.weight_shapes   = None
    self.weights_lengths = None
    self.plots = [] #points for matplotlib
    self.episodes = 0

    self.best_results = {}

  #--- Initialize Population --------------------------------------------------+
  def create_nn(self):
    '''Create individual of population'''

    model = Sequential()
    model.add(Dense(self.num_features, input_shape = (self.num_features,)))
    
    for layer in range(self.num_layers):

        try:
            nodes=self.nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = self.default_nodes

        model.add(Dense(units = nodes, activation = 'relu'))
    
    #output layer
    model.add(Dense(units = self.num_outputs, activation = self.activation))
    model.compile(optimizer = Adam(lr=0.001), loss = 'mse', metrics=['accuracy'])

    #create deserialize dependencies
    if self.weight_shapes is None:
      model.summary()
      self.weight_shapes = []
      self.weights_lengths = []

      weights = model.get_weights()
      for x in weights:
        self.weight_shapes.append(x.shape)

        #generate indicies of weights to recreate weight structure from gene string
        length = len(x.reshape(1, -1)[0].tolist())
        if not self.weights_lengths:
          self.weights_lengths.append(length)
        else:
          self.weights_lengths.append(self.weights_lengths[len(self.weights_lengths)-1]+length)
      if self.mxrt == 'default':
        self.mxrt = math.log(self.weights_lengths[-1], 2)/(self.weights_lengths[-1])
      print('Weight Shapes:', self.weight_shapes)
      print('Weight Lengths:', self.weights_lengths)
      print('Mutation Rate:', self.mxrt)
      print('Crossover Type:', self.cxtype)
      print('Selection Type:', self.selection_type)
      print('Sharpness:', self.sharpness)
    return model
  
  def create_transfer_cnn(self, ref_model=None, fcn_weights=None):
    '''creates resnet model. will load deserialized weights by passing in weights'''

    if not ref_model:
      model = ResNet50(weights='imagenet', include_top=False, input_shape=(self.envs[0].observation_space.shape))
      for layer in model.layers:
        layer.trainable = False
      
      pretrained_weights = model.get_weights()

      flattened = Flatten()(model.output)
      #Add FCN
      for layer in range(self.num_layers):

        try:
            nodes=self.nodes_per_layer[layer]
        except IndexError:
            nodes = None

        if nodes is None:
            nodes = self.default_nodes

        if layer == 0:
          add_layer = Dense(units = nodes, activation = 'relu')(flattened)
        else:
          add_layer = Dense(units = nodes, activation = 'relu')(add_layer)
      
      if self.num_layers:
        output = Dense(units = self.num_outputs, activation = self.activation)(add_layer)
      else:
        output = Dense(units = self.num_outputs, activation = self.activation)(flattened)

      model = Model(model.inputs, output)
      model.compile(Adam(lr=1e-3), 'mse', metrics=['acc'])
    else:
      model = ref_model

      if fcn_weights:
        assert len(fcn_weights) == len(self.weight_shapes), \
          f'Invalid Weight Structure. Expected {len(self.weight_shapes)}, got {len(fcn_weights)}.'
        all_weights = model.get_weights()
        untrainable = all_weights[:-len(self.weight_shapes)]
        weights = all_weights[-len(self.weight_shapes):]
        # print('Deserialized weights length:', len(weights))
        for i, matrix in enumerate(weights):
          # print('Original', matrix)
          matrix[:] = fcn_weights[i]
          # print('Result', matrix)
      
        model.set_weights(untrainable + weights)
    
    #create deserialize dependencies
    if self.weight_shapes is None:
      model.summary()
      self.weight_shapes = []
      self.weights_lengths = []

      weights = model.get_weights()
      self.full_weights_length = len(weights)
      self.pretrained_weights_length = len(pretrained_weights)
      for i in range(len(pretrained_weights), len(weights)):
        self.weight_shapes.append(weights[i].shape)

        #generate indicies of weights to recreate weight structure from gene string
        length = len(weights[i].reshape(1, -1)[0].tolist())
        if not self.weights_lengths:
          self.weights_lengths.append(length)
        else:
          self.weights_lengths.append(self.weights_lengths[len(self.weights_lengths)-1]+length)
      if self.mxrt == 'default':
        self.mxrt = math.log(self.weights_lengths[-1], 10)/(self.weights_lengths[-1])
      print('Weight Shapes:', self.weight_shapes)
      print('Weight Lengths:', self.weights_lengths)
      print('Mutation Rate:', self.mxrt)
      print('Crossover Type:', self.cxtype)
      print('Selection Type:', self.selection_type)
      print('Sharpness:', self.sharpness)
    
    return model
  
  def create_population(self):
    if self.transfer:
      model = self.create_transfer_cnn()
    else:
      model = self.create_nn()
    self.models.append(model)
    model_genes = self.serialize(model)
    for i in range(self.pop_size):
      new_ind = reinitLayers(model)
      self.pop.append(self.serialize(new_ind))
      print('Model', i, 'created.')
    
    print('Correctly generated:', not(False in [len(model_genes) == len(self.pop[i]) for i in range(self.pop_size)]))
  #----------------------------------------------------------------------------+

  #--- Fitness Calculation ----------------------------------------------------+

  def quality(self, genes, i):
    '''
      fitness function. Returns quality of model
      Runs average of self.sharpness episodes of environment
    '''
    print(f'Testing model {i}...', end='')
    #load weights
    model = self.load_weights(genes)
    
    #Test model
    total_rewards = []
    for epoch in range(self.sharpness):
      self.episodes += 1
      done = False
      rewards = []
      envstate = self.envs[0].reset()
      while not done:
        action = self.predict(model, envstate)
        envstate, reward, done, info = self.envs[0].step(action)
        rewards.append(reward)
      
      total_rewards.append(reward)
    
    # if 5 >= self.sharpness >= 1:
    #   result = max(total_rewards)
    # else:
    result = sum(total_rewards)/len(total_rewards)
    print(result)
    return result
  
  #----------------------------------------------------------------------------+
  
  #--- Breed Population -------------------------------------------------------+
  def selection(self):
    '''
      generate mating pool, tournament && elistist selection policy
    '''
    selection = []

    ranked = [] #ranked models, best to worst
    for i, genes in enumerate(self.pop):
      fitness = self.quality(genes, i)
      ranked.append((i, fitness))
      if self.fitness_goal is not None and fitness >= self.fitness_goal:
        #goal met? If so, early stop
        if self.validation_size:
          valid = self.validate(self.models[0])
        else:
          valid = True
        
        if valid:
          self.goal_met = self.models[0] #save model that met goal
          self.best_fit = (i, fitness)
          break

    if not self.goal_met:  #if goal met prepare to terminate
      ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)
      print('Ranked:', ranked)
      self.ranked = ranked
      self.best_fit = ranked[0]

      for i in range(self.elitist):
        selection.append(ranked[i])

      if self.selection_type == 'tour':
        while len(selection) < self.pop_size:
          tourny = random.sample(ranked, self.tour)
          selection.append(max(tourny, key=lambda x:x[1]))

      elif self.selection_type == 'cxrt':
        while len(selection) < self.pop_size:
          for model in random.sample(ranked, len(ranked)):
            if random.random() < self.cxrt:
              selection.append(model)
            

    self.plots.append(self.best_fit)
    if self.best_fit[1] >= self.best_results.get('fitness', -1000000) or self.goal_met:
      self.best_results['fitness'] = self.best_fit[1]
      self.best_results['genes'] = [gene for gene in self.pop[self.best_fit[0]]]
    return selection[:self.pop_size]

  def selection_mp(self):
    '''
      runs processes in parallel
      generate mating pool, tournament && elistist selection policy
    '''
    selection = []

    #create gene dependencies
    all_genes = []
    for i in range(self.pop_size):
      genes = Array('f', range(len(self.pop[i])))
      for j in range(len(self.pop[i])):
        genes[j] = self.pop[i][j]
      all_genes.append(genes)

    nodes_per_layer = Array('f', range(len(self.nodes_per_layer)))
    for j in range(len(self.nodes_per_layer)):
      nodes_per_layer[j] = self.nodes_per_layer[j]

    fitnesses = Array('f', range(self.pop_size))
    processed = 0
    processes = []
    while processed < self.pop_size:
      if processed < self.pop_size - len(processes) and  len(processes) < self.cores:
        i = len(processes) + processed

        genes = all_genes[i]
        
        obj = {
          'index':      i,
          'genes':      genes,
          'res':        fitnesses, 
          'env':        self.envs[len(processes)], 
          'layers':     self.num_layers,
          'transfer':   self.transfer,
          'outputs':    self.num_outputs,
          'inputs':     self.num_features,
          'sharpness':  self.sharpness,
          'shapes':     self.weight_shapes,
          'activation': self.activation,
          'lengths':    self.weights_lengths,
          'nodes_per_layer': nodes_per_layer,
        }

        # print('Restructred Genes', obj['index'], [val for val in genes], '\n\n\n')
        p = Process(target=multi_quality, kwargs=obj)
        processes.append(p)
        print('Starting process', i)
        p.start()
      
      #remove completed processes
      ind = 0
      while ind < len(processes):
        p = processes[ind]
        if not p.is_alive():
          #terminate process
          p.join()
          processes.pop(ind)
          ind -= 1

          processed += 1
          self.episodes += self.sharpness
        ind += 1

    ranked = [] #ranked models, best to worst
    results = [val for val in fitnesses]
    for i, fitness in enumerate(results):
      ranked.append((i, fitness))

    ranked = sorted(ranked, key=operator.itemgetter(1), reverse=True)
    print('Ranked:', ranked)
    self.ranked = ranked
    self.best_fit = ranked[0]
    
    for model in ranked: #model = (i, fitness)
      if self.fitness_goal is not None and model[1] >= self.fitness_goal:
        #goal met? If so, early stop
        i = model[0] #model number
        if self.validation_size:
          valid = self.validate(self.load_weights(self.pop[i]))
        else:
          valid = True
        
        if valid:
          self.goal_met = self.models[0] #save model that met goal
          self.best_fit = model
          break

    if not self.goal_met:  #if goal met prepare to terminate
      for i in range(self.elitist):
        selection.append(ranked[i])

      if self.selection_type == 'tour':
        while len(selection) < self.pop_size:
          tourny = random.sample(ranked, self.tour)
          selection.append(max(tourny, key=lambda x:x[1]))

      elif self.selection_type == 'cxrt':
        while len(selection) < self.pop_size:
          for model in random.sample(ranked, len(ranked)):
            if random.random() < self.cxrt:
              selection.append(model)
            

    self.plots.append(self.best_fit)
    if self.best_fit[1] >= self.best_results.get('fitness', -1000000) or self.goal_met:
      self.best_results['fitness'] = self.best_fit[1]
      self.best_results['genes'] = [gene for gene in self.pop[self.best_fit[0]]]
    return selection[:self.pop_size]

  def crossover(self, parents):
    children = [] #gene strings

    #keep elites
    for i in range(self.elitist):
      index = parents[i][0]
      children.append(self.pop[index])

    parents = random.sample(parents, len(parents)) #randomize breeding pool

    #breed rest
    i = 0 #parent number, genes to get
    while len(children) < self.pop_size:
      parent1 = parents[i]
      parent2 = parents[len(parents)-i-1]

      parent1_genes = self.pop[parent1[0]]
      parent2_genes = self.pop[parent2[0]]
      if self.cxtype == 'splice':
        if self.num_layers > 1:
          genes = []
          for index, len_ in enumerate(self.weights_lengths): #splice each layer
            if index == 0:
              range_ = (0, len_)
            else:
              range_ = (self.weights_lengths[index-1], len_)

            #splice genes
            start = range_[0]
            end = range_[1]
            geneA = random.randrange(start, end)
            geneB = random.randrange(geneA, end+1)
            geneA -= start
            geneB -= start

            genes.append(splice_list(parent1_genes[start:end], parent2_genes[start:end], geneA, geneB))
          child = flatten(genes)
        else:
          geneA = random.randrange(0, len(parent1_genes))
          geneB = random.randrange(geneA, len(parent1_genes)+1)

          child = splice_list(parent1_genes, parent2_genes, geneA, geneB)
      elif self.cxtype == 'weave':
        child = [] #gene string
        for j in range(len(parent1_genes)):
          if j % 2 == 0:
            child.append(parent1_genes[j])
          else:
            child.append(parent2_genes[j])
      else:
        child = ((np.array(parent1_genes) + np.array(parent2_genes)) / 2).tolist()
      
      children.append(child)
      i+=1
    
    return children
  
  def mutate(self, population):
    if self.mx_type!='default':
      '''randomize layers, VERY INEFFICIENT'''
      for ind, individual in enumerate(population):
        mutated = False
        if ind >= len(population) - self.random_children: #Randomly initialize last child
          model = reinitLayers(self.models[0])
          mutated = True
        else:
          model = self.load_weights(individual)
          session = backend.get_session()
          for layer in model.layers:
            if layer.trainable:
              if random.random() < self.mxrt:
                if not mutated:
                  mutated = True
                for v in layer.__dict__:
                  v_arg = getattr(layer,v)
                  if hasattr(v_arg,'initializer'):
                    initializer_method = getattr(v_arg, 'initializer')
                    initializer_method.run(session=session)
        if mutated:
          weights = self.serialize(model)
          for i, gene in enumerate(individual):
            individual[i] = weights[i]

    else:
      mxrt = self.mxrt
      ref_genes = self.serialize(reinitLayers(self.models[0]))
      for ind, individual in enumerate(population):
        #if ind >= self.elitist: #ignore elites
          if self.random_children and mxrt != 1:
            if ind >= len(population) - self.random_children: #Randomly initialize last child
              mxrt = 1
          if mxrt == 1: #for random children
            ref_genes = self.serialize(reinitLayers(self.models[0]))
          for i, gene in enumerate(individual):
            if mxrt == 1 or random.random() < mxrt:
              individual[i] = ref_genes[i]
    return population
  #----------------------------------------------------------------------------+
  
  #--- Train/Evaluate ---------------------------------------------------------+

  def train(self, filename=None, target='best_model.h5', callbacks=None):
    self.create_population()
    print('Population created', len(self.pop))

    if filename:
      self.models[0].load_weights(filename)
      self.pop[0] = self.serialize(self.models[0])
      print('Model loaded from', filename)

    self.start_time = datetime.datetime.now()
    print(f'Starting at: {self.start_time}')

    for i in range(self.generations):
      try:
        print('\nGeneration:', i+1, '/', self.generations)
        if self.cores > 1:
          parents = self.selection_mp()
        else:
          parents = self.selection()

        if not(i == self.generations - 1): #dont perform mutatations on last gen
          print('Goal not met. Parents selected.')
          print('Best fit:', self.best_fit)
          print('Best Results', self.best_results.get('fitness'))
          children = self.crossover(parents)
          print('Breeding done.')
          new_pop = self.mutate(children)
          print('Mutations done.')
          
          print('New pop:', len(new_pop))
          self.pop = new_pop
      finally:
        if callbacks:
          for func in callbacks:
            func()
      dt = datetime.datetime.now() - self.start_time
      print('Time Running: ', format_time(dt.total_seconds()))
      if self.goal_met:
        print(f'Goal met! Episodes: {self.episodes}')
        break
    
    self.save_best(target=target)


  def evaluate(self, filename=None, epochs=0):
    if self.goal_met or filename:
      #load model
      if filename:
        if self.transfer:
          model = self.create_transfer_cnn()
        else:
          model = self.create_nn()
        model.load_weights(filename)
        print(f'Weights loaded from {filename}')
      else:
        model = self.goal_met

      epoch = 0
      total_rewards = []
      #display results
      while (True, epoch<epochs)[epochs>0]:
        done = False
        rewards = []
        envstate = self.envs[0].reset()
        while not done:
          action = self.predict(model, envstate)
          envstate, reward, done, info = self.envs[0].step(action)
          if not epochs:
            self.envs[0].render()
          rewards.append(reward)

        print('Reward:', sum(rewards))
        total_rewards.append(sum(rewards))
        rewards = []
        epoch+=1
      print('Epochs:', epoch, 'Average reward:', sum(total_rewards)/len(total_rewards))
  #----------------------------------------------------------------------------+

  #--- Validate Fitness -------------------------------------------------------+
  def validate(self, model):
    print('Validating Model...', end='')
    
    total_rewards = []
    n_epochs = self.validation_size
    #test results
    for epoch in range(n_epochs):
      done = False
      rewards = []
      envstate = self.envs[0].reset()
      while not done:
        action = self.predict(model, envstate)
        envstate, reward, done, info = self.envs[0].step(action)
        rewards.append(reward)

      total_rewards.append(sum(rewards))
    print(sum(total_rewards)/len(total_rewards))
    return sum(total_rewards)/len(total_rewards) >= self.fitness_goal
  #----------------------------------------------------------------------------+

  #--- Graph Functions --------------------------------------------------------+

  def show_plot(self):
    y = [self.plots[i][1] for i in range(len(self.plots))] #best fitness
    x = [i for i in range(len(self.plots))] #generation

    plt.plot(x, y, label='Best fitness')
    plt.legend(loc=4)
    plt.show()
    
  #----------------------------------------------------------------------------+

  #--- Helper Functions -------------------------------------------------------+

  def save_best(self, target='best_model.h5'):
    if self.best_results['fitness']:
      genes = self.best_results['genes']
    elif self.best_fit:
      genes = self.pop[self.best_fit[0]]
    model = self.load_weights(genes)
    model.save_weights(target)
    print(f'Best results saved to {target}')

  def predict(self, model, envstate):
    ''' decide best action for model. '''
    qvals = model.predict(self.adj_envstate(envstate))[0] 
    if self.num_outputs == 1:
      action = qvals #continuous action space
    else:
      action = np.argmax(qvals) #discrete action space
    
    return action

  def adj_envstate(self, envstate):
    if self.transfer:
      return np.expand_dims(envstate, axis=0)
    return envstate.reshape(1, -1)

  def serialize(self, model):
    '''
      serializes model's weights into a gene string
    '''
    
    if self.transfer:
        weights = model.get_weights()[-len(self.weight_shapes):]
    else:
        weights = model.get_weights()
    flattened = []
    for arr in weights:
      flattened+=arr.reshape(1, -1)[0].tolist()
    
    return flattened

  def deserialize(self, genes):
    '''
      deserializes gene string into weights
    '''
    shapes = self.weight_shapes
    lengths = self.weights_lengths
    
    weights = []
    for i, val in enumerate(lengths):
      if i == 0:
        begin = 0
      else:
        begin = lengths[i-1]
      weights.append(np.array(genes[begin:val]).reshape(shapes[i]))
    
    return weights

  def load_weights(self, genes):
    if self.transfer:
      model = self.models[0]
      self.models[0] = self.create_transfer_cnn(\
        ref_model=model, fcn_weights=self.deserialize(genes)
      )
    else: 
        self.models[0].set_weights(self.deserialize(genes))
    
    return self.models[0]

def splice_list(list1, list2, index1, index2):
  '''
    combined list1 and list2 taking splice from list1 with starting index `index1`
    and ending index `index2`
  '''
  if index1 == 0:
    splice = list1[index1:index2+1]
    splice += list2[index2+1:len(list1)]
  else:
    splice = list2[:index1] + list1[index1:index2+1]
    splice += list2[index2+1:len(list1)]
  
  return splice

def flatten(L):
  'flatten 2d list'
  flat = []
  for l in L:
    flat += l
  
  return flat

# This is a small utility for printing readable time strings:
def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)

def reinitLayers(model):
  session = backend.get_session()
  for layer in model.layers:
    if layer.trainable:
      for v in layer.__dict__:
          v_arg = getattr(layer,v)
          if hasattr(v_arg,'initializer'):
              initializer_method = getattr(v_arg, 'initializer')
              initializer_method.run(session=session)
              # print('reinitializing layer {}.{}'.format(layer.name, v))
  return model
#------------------------------------------------------------------------------+