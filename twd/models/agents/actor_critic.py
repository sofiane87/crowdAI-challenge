from util.util import *
import numpy as np
import sys
import os
from rl.agents import DDPGAgent

from keras_model import keras_model


class actor_critic(keras_model):


	def __init__(self,shared_object):

		log_info('building actor-critic specific parameters ')
		
		self.gamma = shared_object.get('gamma',.99)
		self.nb_steps_warmup_critic = shared_object.get('nb_steps_warmup_critic',100)
		self.nb_steps_warmup_actor = shared_object.get('nb_steps_warmup_actor',100)
		self.target_model_update =  shared_object.get('target_model_update',1.0e-3)
		self.delta_clip = shared_object.get('delta_clip',1.)

		log_info('loading actor critic specific parameters is done')
		
		super(self.__class__,self).__init__(shared_object)



	def build_agent(self):
		### Building Agent
		log_info("building DDPGAgent ...") 
		self.agent = DDPGAgent(nb_actions=self.nb_actions, actor=self.networks.actor, critic=self.networks.critic, critic_action_input=self.networks.action_input,
                  memory=self.networks.memory, nb_steps_warmup_critic=self.nb_steps_warmup_critic, nb_steps_warmup_actor=self.nb_steps_warmup_actor,
                  random_process=self.networks.random_process, gamma=self.gamma, target_model_update=self.target_model_update,
                  delta_clip=self.delta_clip)

		# self.agent = DDPGAgent(nb_actions=self.nb_actions, actor=self.networks.actor, critic=self.networks.critic, critic_action_input=self.networks.action_input,
  #                 memory=self.networks.memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
  #                 random_process=self.networks.random_process, gamma=.99, target_model_update=1e-3,
  #                 delta_clip=1.)


		log_info("Adding optimizer ...")
		self.agent.compile(self.optimizer, self.metrics)
		log_info("Optimizer added.")


