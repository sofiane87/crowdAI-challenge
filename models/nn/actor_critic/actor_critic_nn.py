
### The script should contain this three functions 
class actor_critic_nn:
	def build_actor(self):
		actor = None
		raise NotImplementedError
		return actor

	def build_critic(self):
		critic_input = None
		critic = None
		raise NotImplementedError
		return critic, critic_input

	def build_memory(self):
		memory = None
		raise NotImplementedError
		return critic, critic_input

	def build_random_process(self,env):
		random_process = None
		raise NotImplementedError
		return random_process