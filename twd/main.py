import argparse
from util.util import *


### Getting all arguments

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--config', dest='config', action='store', default="./config/keras_config.yaml")
args = parser.parse_args()

log_info('loading shared object ...')
shared_object = parse_yaml(args.config)
log_info('building model ...')
model= build_model(shared_object)
log_info('running the model ...')
model.run()

