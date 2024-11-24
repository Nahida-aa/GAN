import json
import os
        
        
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
# 加载配置
config_path = os.path.join(CONFIG_DIR, 'config.json')
with open(config_path) as config_file:
    config_dict = json.load(config_file)

DATA_DIR = os.path.join(BASE_DIR, config_dict['path']['data_dir'])
IMAGES_DIR = os.path.join(BASE_DIR, config_dict['path']['images_dir'])
WEIGHTS_DIR = os.path.join(BASE_DIR, config_dict['path']['weights_dir'])
LOGS_DIR = os.path.join(BASE_DIR, config_dict['path']['logs_dir'])


os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRAINING_PARAMETERS = config_dict["training_parameters"]
BATCH_SIZE = TRAINING_PARAMETERS['batch_size']
LEARNING_RATE = TRAINING_PARAMETERS['learning_rate']
START_EPOCH = TRAINING_PARAMETERS['start_epoch']
END_EPOCH = TRAINING_PARAMETERS['end_epoch']
SAVE_INTERVALS = TRAINING_PARAMETERS['save_intervals']