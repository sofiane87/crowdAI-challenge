# Derived from keras-rl
import opensim as osim
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *
from keras.optimizers import RMSprop
from keras_model import keras_model 

class actor_critic(keras_model):

	def __init__(self,name='actor_critic',env='example',network_name='example',gamma=.99,target_model_update=1e-3,delta_clip=1.,nb_steps_warmup_actor=100,nb_steps_warmup_critic=100,visualize=False,lr=.001, clipnorm=1,metrics=['mae']):

		### Adding inputs to object		
		if env == 'example':
			env = RunEnv(visualize)
			env.reset()
		
		self.env = env
		self.model_name = name

		actor, critic, memory, action_input, random_process = self.get_network(network_name)

		self.actor = actor
		self.critic = critic
		self.memory = memory
		self.action_input = action_input
		self.random_process = random_process
		self.gamma = gamma
		self.target_model_update = target_model_update
		self.delta_clip = delta_clip
		self.nb_actions = env.action_space.shape[0]
		self.nb_steps_warmup_critic = nb_steps_warmup_critic
		self.nb_steps_warmup_actor = nb_steps_warmup_actor
		self.visualize = visualize
		self.lr = lr
		self.clipnorm = clipnorm
		self.metrics = metrics
		self.nb_max_episode_steps= env.timestep_limit

		### Building Agent 
		self.agent = DDPGAgent(nb_actions=self.nb_actions, actor=self.actor, critic=self.critic, critic_action_input=self.action_input,
                  memory=self.memory, nb_steps_warmup_critic=self.nb_steps_warmup_critic, nb_steps_warmup_actor=self.nb_steps_warmup_actor,
                  random_process=self.random_process, gamma=self.gamma, target_model_update=self.target_model_update,
                  delta_clip=self.delta_clip)

		self.agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])


	def populate_missing_input(self,env, actor, critic, memory, action_input, random_process,visualize=False):

		### building 


		if actor == 'example':
			actor = Sequential()
			actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
			actor.add(Dense(32))
			actor.add(Activation('relu'))
			actor.add(Dense(32))
			actor.add(Activation('relu'))
			actor.add(Dense(32))
			actor.add(Activation('relu'))
			actor.add(Dense(nb_actions))
			actor.add(Activation('sigmoid'))

		if critic == 'example':
			action_input = Input(shape=(nb_actions,), name='action_input')
			observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
			flattened_observation = Flatten()(observation_input)
			x = concatenate([action_input, flattened_observation])
			x = Dense(64)(x)
			x = Activation('relu')(x)
			x = Dense(64)(x)
			x = Activation('relu')(x)
			x = Dense(64)(x)
			x = Activation('relu')(x)
			x = Dense(1)(x)
			x = Activation('linear')(x)
			critic = Model(inputs=[action_input, observation_input], outputs=x)


		if memory == 'example':
			memory = SequentialMemory(limit=100000, window_length=1)

		if random_process == 'example':
			random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)

		return env, actor, critic, memory, action_input, random_process

	def get_network(self, network_name):
		network = self.import_class('nn.' + self.model_name + '.' + network_name )
		network = network()
		actor = network.build_actor(self)
		critic, action_input = network.build_critic(self)
		memory = network.build_memory(self)
		random_process = network.build_random_process(self)
		return actor, critic, memory, action_input, random_process

	def import_class(self,name):
	    components = name.split('.')
	    mod = __import__(components[0])
	    for comp in components[1:]:
	        mod = getattr(mod, comp)
	    return mod



print('starting')
test_agent = actor_critic()
print('training')
test_agent.train()