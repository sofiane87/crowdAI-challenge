import sys 
import argparse
from util import import_class
from osim.env import *


keras_callbacklist = ['TestLogger','TrainEpisodeLogger','TrainIntervalLogger','FileLogger','Visualizer','ModelIntervalCheckpoint']

### Getting all arguments

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')

parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false', default=True)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--save', dest='save', action='store_true', default=True)
parser.add_argument('--model_class', dest='model_class', action='store', default='actor_critic')
parser.add_argument('--model_name', dest='model_name', action='store', default='actor_critic')
parser.add_argument('--network', dest='network', action='store', default='example')
parser.add_argument('--submit',dest='submit',action='store_false',default=False)

args = parser.parse_args()


### Starting the Environment 

env = RunEnv(args.visualize)
env.reset()

### Building the model

model_class = import_class('models.'+ args.model_class)
model = model_class(name=args.model_name,env=env,network_name=args.network)

### Training the model / Loading The model 
if args.train : 
	model.train()
	if args.save:
		model.save()
else:
	model.load()

### Testing
if args.test():
	model.test()

### submitting 
if args.submit:
	model.submit()