import os
import configparser

config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
if not os.path.isfile(config_path):
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.ini')
config: configparser.ConfigParser = configparser.ConfigParser()
config.read(config_path)
