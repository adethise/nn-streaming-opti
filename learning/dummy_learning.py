#!/usr/bin/env python3

import numpy as np

class Learner:

    def __init__(self, sess, a_dims, s_lengths, nn_model):
        self.a_dims = a_dims
        self.s_lengths = s_lengths
        print('a_dims:', a_dims)
        print('s_lengths:', s_lengths)

        self.s_dims = len(self.s_lengths), max(self.s_lengths)

        self.entropy_record = [0]


    def predict(self, metrics):
        assert len(metrics) == len(self.s_lengths), 'Wrong number of metrics'

        # Copy the metrics to a fixed-size np.array
        state = np.zeros(self.s_dims)

        for i, metric in enumerate(metrics):
            assert self.s_lengths[i] == len(metric), \
                    f'Metric {i} has the wrong length (expected {self.s_lengths[i]}, got {len(metric)})'
            state[i, :len(metric)] = metric

        print('State:', state)

        actions = list()
        for a_dim in self.a_dims:
            actions.append(a_dim - 1)

        print('Actions:', actions)

        return actions


    def train(self, actions, metrics, reward, epoch):
        action_vecs = [np.zeros(a_dim) for a_dim in self.a_dims]
        for i, action in enumerate(actions):
            action_vecs[i][action] = 1

        print(action_vecs)
