import time
import random

class Simulator:

    def __init__(self, random_seed = 42):
        random.seed(random_seed)

    def get_performance(self, action):

        if action == 'A':
            return random.random() * 100, 100, 100
        elif action == 'B':
            return random.random() * 200, 100, 100
        else:
            return random.random() * 300, 100, 100
