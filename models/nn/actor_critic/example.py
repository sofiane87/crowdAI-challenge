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

import argparse
import math
from actor_critic_nn import actor_critic_nn

class example(actor_critic_nn):

	### The script should contain this three functions 
	def build_actor(self,parent_object):
		env = parent_object.env
		nb_actions = env.action_space.shape[0]
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
		return actor

	def build_critic(self,parent_object):
		env = parent_object.env
		nb_actions = env.action_space.shape[0]
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
		return critic, action_input

	def build_memory(self,parent_object):
		env = parent_object.env
		memory = SequentialMemory(limit=100000, window_length=1)
		return memory

	def build_random_process(self,parent_object):
		env = parent_object.env
		random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.2, size=env.noutput)
		return random_process
