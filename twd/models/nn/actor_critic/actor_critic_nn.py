### The script should contain this three functions 
class actor_critic_nn(object):
	def __init__(self,shared_object):
		self.shared_object = shared_object
		self.build()

	def build(self):
		self.build_action_input()
		self.build_actor()
		self.build_critic()		
		self.build_memory()
		self.build_random_process()


	def build_actor(self):
		raise NotImplementedError

	def build_critic(self):

		raise NotImplementedError

	def build_memory(self):
		raise NotImplementedError


	def build_random_process(self):
		raise NotImplementedError


	def build_action_input(self):
		raise NotImplementedError
