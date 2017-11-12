import time

class Simulator:

    def get_performance(self, action):

        if action == 'A':
            return 100, 100, 100
        elif action == 'B':
            return 200, 100, 100
        else:
            return 300, 100, 100
