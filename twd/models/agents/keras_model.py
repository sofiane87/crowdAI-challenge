# Derived from keras-rl
import numpy as np
import sys
import os

import opensim as osim
from osim.env import RunEnv
from osim.http.client import Client

from util.util import *


class keras_model(object):
	
	def __init__(self,shared_object):
		
		log_info('setting keras_model main parameters')
		self.shared_object = shared_object
		self.model_class = shared_object.get('model_class',None)
		self.name = shared_object.get('model_name',None)
		self.network_name = shared_object.get('network',None)
		self.train_bool = shared_object.get('train',True)
		self.test_bool = shared_object.get('test',True)
		self.load_bool = not(self.train_bool)
		self.submit_bool = shared_object.get('submit',False)
		self.tokenId = shared_object.get('submit_token',None)
		self.visualize = shared_object.get('visualize',False)
		self.save_bool = shared_object.get('save',True)
		self.save_path = shared_object.get('save_path',os.path.join('model_weights',self.model_class,self.name,self.network_name +'.h5f'))
		self.save_folder = os.path.dirname(self.save_path)
		self.env = shared_object.get('env',None)
		if self.env == None:
			self.env = RunEnv(self.visualize)
			self.shared_object['env'] = self.env
			shared_object['env'] = self.env
			
		self.env.reset()
		self.nb_actions = self.env.action_space.shape[0]
		self.metrics = shared_object.get('metrics',['mae'])
		self.optimizer_name = shared_object.get('optimizer','Adam')
		self.optimizer_params = shared_object.get('optimizer_params',None)

		log_info("setting keras_model's training parameters")
		self.train_parameters = {}
		self.train_parameters['nb_steps'] = shared_object.get('nb_steps',0)
		self.train_parameters['action_repetition'] = shared_object.get('action_repetition',0)
		self.train_parameters['callback_names'] = shared_object.get('callback_names',None)
		self.train_parameters['callbacks'] = load_callbacks(self.train_parameters['callback_names'])
		self.train_parameters['verbose'] = shared_object.get('verbose',0)
		self.train_parameters['nb_max_start_steps'] = shared_object.get('nb_max_start_steps',0)
		self.train_parameters['start_step_policy'] = shared_object.get('start_step_policy',None)
		self.train_parameters['log_interval'] = shared_object.get('log_interval',1)
		self.train_parameters['nb_max_episode_steps'] = shared_object.get('nb_max_episode_steps',self.env.timestep_limit)

		log_info("setting keras_model's testing parameters")
		
		self.test_parameters = {}
		self.test_parameters['nb_episodes'] = shared_object.get('test_nb_episodes',1)
		self.test_parameters['nb_max_episode_steps'] = shared_object.get('test_nb_max_episode_steps',1)


		log_info('loading networks : {}'.format(self.network_name))
		self.load_networks()
		log_info('loading networks done')


		log_info('building optimizer : {}'.format(self.optimizer_name))
		self.build_optimizer()
		log_info('optimizer built sucessfully')

		log_info('building the agent')
		self.build_agent()
		log_info('agent sucessfully built')



	def train(self):
		
		"""
		# Arguments
            nb_steps (integer): Number of training steps to be performed.
            action_repetition (integer): Number of times the agent repeats the same action without
                observing the environment again. Setting this to a value > 1 can be useful
                if a single action only has a very small effect on the environment.
            callbacks (list of `keras.callbacks.Callback` or `rl.callbacks.Callback` instances):
                List of callbacks to apply during training. See [callbacks](/callbacks) for details.
            verbose (integer): 0 for no logging, 1 for interval logging (compare `log_interval`), 2 for episode logging
            visualize (boolean): If `True`, the environment is visualized during training. However,
                this is likely going to slow down training significantly and is thus intended to be
                a debugging instrument.
            nb_max_start_steps (integer): Number of maximum steps that the agent performs at the beginning
                of each episode using `start_step_policy`. Notice that this is an upper limit since
                the exact number of steps to be performed is sampled uniformly from [0, max_start_steps]
                at the beginning of each episode.
            start_step_policy (`lambda observation: action`): The policy
                to follow if `nb_max_start_steps` > 0. If set to `None`, a random action is performed.
            log_interval (integer): If `verbose` = 1, the number of steps that are considered to be an interval.
            nb_max_episode_steps (integer): Number of steps per episode that the agent performs before
                automatically resetting the environment. Set to `None` if each episode should run
                (potentially indefinitely) until the environment signals a terminal state.
        # Returns
            A `keras.callbacks.History` instance that recorded the entire training process.
        """

		callback_history = self.agent.fit(self.env, nb_steps=self.train_parameters['nb_steps'], action_repetition=self.train_parameters['action_repetition'], callbacks=self.train_parameters['callbacks'], verbose=self.train_parameters['verbose'],
            visualize=self.visualize, nb_max_start_steps=self.train_parameters['nb_max_start_steps'], start_step_policy=self.train_parameters['start_step_policy'], log_interval=self.train_parameters['log_interval'],
            nb_max_episode_steps=self.train_parameters['nb_max_episode_steps'])

		if self.save_bool:
			if not(os.pasth.exists(self.save_folder)):
				os.makedirs(self.save_folder)
			agent.save_weights(self.savePath, overwrite=True)

		return callback_history

	def load(self):
		log_info('loading model : {}'.format(self.name))
		self.agent.load_weights(self.save_path)

	def test(self):
	    self.agent.test(self.env, nb_episodes=self.test_parameters['nb_episodes'], visualize=self.test_parameters['visualize'], nb_max_episode_steps=self.test_parameters['nb_max_episode_steps'])

	def submit(self):
		

		remote_base = 'http://grader.crowdai.org:1729'
		env = RunEnv(visualize=self.visualize)
		client = Client(remote_base)

		# Create environment
		observation = client.env_create(self.submit_token)

		# Run a single step
		#
		# The grader runs 3 simulations of at most 1000 steps each. We stop after the last one
		while True:
		    [observation, reward, done, info] = client.env_step(self.agent.forward(observation))
		    if done:
		        observation = client.env_reset()
		        if not observation:
		            break

		client.submit()


	def build_optimizer(self):
		log_info('loading optimizer class : {}'.format(self.optimizer_name))
		optimizer_class = import_class('keras.optimizers.{}'.format(self.optimizer_name))
		self.optimizer = optimizer_class(**self.optimizer_params)


	def run(self):
		if self.train_bool:
			log_info('starting to train the model...')
			self.train()
		if self.load_bool:
			log_info('starting to load the model...')
			self.load()
		if self.test_bool:
			log_info('starting testing the model ...')
			self.test()
		if self.submit_bool:
			log_info('starting to submit the model')
			self.submit()

	def load_networks(self):
		network_class = import_class('models.nn.{}.{}'.format(self.model_class,self.network_name))	
		self.networks  = network_class(self.shared_object)

	def build_agent(self):
		raise NotImplementedError
