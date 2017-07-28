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

	def __init__(self,name='actor_critic',env='example',network_name='example',gamma=.99,target_model_update=1e-3,delta_clip=1.,nb_steps_warmup_actor=100,nb_steps_warmup_critic=100,visualize=False,lr=.001, clipnorm=1,metrics=['mae'],save_path=None):

		### Adding inputs to object		
		if env == 'example':
			env = RunEnv(visualize)
			env.reset()
		
		self.env = env
		self.model_name = name

		actor, critic, memory, action_input, random_process = self.get_network(network_name)

		self.network_name = network_name
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



