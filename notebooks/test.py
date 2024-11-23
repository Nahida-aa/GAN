import os
import sys

import src
print(src.__package__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(f"BASE_DIR: {BASE_DIR}")
sys.path.append(BASE_DIR)
print(sys.path)
import src.config
print(src.config.__package__)
print(dir(src))
# from src.config import *
# from ..src import DATA_DIR
