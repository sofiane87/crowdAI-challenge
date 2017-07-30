import yaml
import os
# import logging
import sys 
import opensim as osim
from osim.env import RunEnv
import time
import inspect



# FORMAT = '%(asctime)-8s - %(name)s - %(message)s'
# logging.basicConfig(format=FORMAT)

	
def log_info (message): 

	stack = inspect.stack()
	try :
		the_class = stack[1][0].f_locals["self"].__class__.__name__
		the_method = stack[1][0].f_code.co_name
		
		if '__init__' not in the_method:
			name = the_class + '.' + the_method
		else:
			name = the_class
	except:
		name = 'main'
	current_time = time.strftime('%Y-%b-%d %l:%M%p,%Z')
	print('{} - {} - {}'.format(current_time,name,message))
	

# logger.setLevel(logging.INFO)

def import_class(name):
	components = name.split('.')
	mod = __import__(components[0])
	for comp in components[1:]:
		mod = getattr(mod, comp)
	return mod

def parse_yaml(file_name):
	with open(file_name) as f:
		# use safe_load instead load
		dataMap = yaml.load(f)
	parent_config = dataMap.get('parent_config',None)
	if parent_config != None:
		shared_object = parse_yaml(parent_config)
	else:
		shared_object = {}

	for key in dataMap:
		shared_object[key] = dataMap.get(key)

	return shared_object

def build_model(shared_object):
	shared_object['env'] = RunEnv(shared_object.get('visualize',False))
	model_class_name = 'models.agents.' + shared_object.get('model_class',None)
	log_info('importing class : {}'.format(model_class_name))
	model_class = import_class(model_class_name)
	log_info('{} successfuly imported'.format(model_class_name))
	log_info('building model')
	model = model_class(shared_object)
	return model

def load_callbacks(callback_names_list):
	keras_callbacklist = ['TestLogger','TrainEpisodeLogger','TrainIntervalLogger','FileLogger','Visualizer','ModelIntervalCheckpoint']
	callbacks = []
	if callback_names_list == None:
		callback_names_list = []

	for callback_name in callback_names_list:
		if callback_name in keras_callbacklist:
			callback_class = import_class('rl.callbacks.{}'.callback_name)
		else:
			callback_class = import_class('callbacks.{}'.callback_name)

		callbacks += callback_class()



