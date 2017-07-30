# Derived from keras-rl
import numpy as np
import sys

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import RMSprop
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import opensim as osim
from osim.env import RunEnv

from actor_critic_nn import actor_critic_nn
from util.util import *

class example(actor_critic_nn):

	def __init__(self,shared_object):
		self.env = shared_object.get("env",None)

		if self.env:
			self.env = RunEnv(shared_object.get('visualize',False))


		self.nb_actions = self.env.action_space.shape[0]
		
		## memory parameters
		self.memoryLimit = shared_object.get('memoryLimit',100000)
		self.window_length = shared_object.get('window_length',1)

		## random process parameters
		self.random_process_theta = shared_object.get('random_process_theta',.15)
		self.random_process_mu = shared_object.get('random_process_mu',0.)
		self.random_process_sigma = shared_object.get('random_process_sigma',.2)

		## building the networks

		super(example,self).__init__(shared_object)

	### The script should contain this three functions 
	def build_actor(self):		
		log_info('building actor network')
		actor = Sequential()
		actor.add(Flatten(input_shape=(1,) + self.env.observation_space.shape))
		actor.add(Dense(32))
		actor.add(Activation('relu'))
		actor.add(Dense(32))
		actor.add(Activation('relu'))
		actor.add(Dense(32))
		actor.add(Activation('relu'))
		actor.add(Dense(self.nb_actions))
		actor.add(Activation('sigmoid'))
		log_info('actor network built')
		self.actor = actor


	def build_action_input(self):
		log_info('building action_input network')
		action_input = Input(shape=(self.nb_actions,), name='action_input')
		self.action_input = action_input
		log_info('action_input network built')


	def build_critic(self):

		log_info('building critic network')
		observation_input = Input(shape=(1,) + self.env.observation_space.shape, name='observation_input')
		flattened_observation = Flatten()(observation_input)
		x = concatenate([self.action_input, flattened_observation])
		x = Dense(64)(x)
		x = Activation('relu')(x)
		x = Dense(64)(x)
		x = Activation('relu')(x)
		x = Dense(64)(x)
		x = Activation('relu')(x)
		x = Dense(1)(x)
		x = Activation('linear')(x)
		self.critic = Model(inputs=[self.action_input, observation_input], outputs=x)

		log_info('critic network built')

	def build_memory(self):

		log_info('building memory network')
		env = self.env
		self.memory = SequentialMemory(limit=self.memoryLimit, window_length=self.window_length)

		log_info('memory network built')

	def build_random_process(self):
		log_info('building random process')
		env = self.env
		self.random_process = OrnsteinUhlenbeckProcess(theta=self.random_process_theta, mu=self.random_process_mu, sigma=self.random_process_sigma, size=self.env.noutput)
		log_info('random process built')

