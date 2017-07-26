# Derived from keras-rl
import opensim as osim
import numpy as np
import sys
import os

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate
from keras.optimizers import Adam

import numpy as np

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

from osim.env import *

from keras.optimizers import RMSprop


class keras_model:
	def __init__(self):
		self.agent = None
		self.env = None
		self.visualize = False
		self.model_name = 'keras_model'
		self.nb_max_episode_steps = 0

	def train(self, nb_steps=10000, action_repetition=1, callbacks=None, verbose=1, nb_max_start_steps=0, start_step_policy=None, log_interval=10000,save_weights=True):
		
		self.agent.fit(self.env, nb_steps=nb_steps, action_repetition=action_repetition, callbacks=callbacks, verbose=verbose,
            visualize=self.visualize, nb_max_start_steps=nb_max_start_steps, start_step_policy=start_step_policy, log_interval=log_interval,
            nb_max_episode_steps=self.nb_max_episode_steps)

		if save_weights == True:
			savePath = os.path.join('model_weights',self.model_name,self.model_name +'.h5f')
			if not(os.path.exists(os.path.join('model_weights',self.model_name))):
				os.makedirs(os.path.join('model_weights',self.model_name))

			agent.save_weights(savePath, overwrite=True)

	def load(self):
		self.agent.load_weights(os.path.join('model_weights',self.model_name,self.model_name +'.h5f'))

	def test(self,env,nb_episodes=1,nb_max_episode_steps=500):
	    self.agent.test(env, nb_episodes=nb_episodes, visualize=self.visualize, nb_max_episode_steps=nb_max_episode_steps)

