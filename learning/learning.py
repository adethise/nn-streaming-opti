#!/usr/bin/env python3

import logging
import os
import tensorflow as tf
import numpy as np

import a3c

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

SUMMARY_DIR = './learning_results'


class Learner:
    '''
    Holds the ML component of our system with actor-critic structure and
    predictor interface.
    '''

    def __init__(self, sess, a_dims, s_lengths, nn_model):
        '''
        Initialize the learner.

        :a_dim: array containing the dimension space for each action
        :s_info: the number of different metric information
        :s_lengths: array containing the length of each metric information
        '''
        if not os.path.exists(SUMMARY_DIR):
            os.makedirs(SUMMARY_DIR)

        self.sess      = sess
        self.a_dims    = a_dims
        self.s_lengths = s_lengths

        self.s_dims = len(self.s_lengths), max(self.s_lengths)

        self.actor  = a3c.ActorNetwork(self.sess, self.a_dims, self.s_lengths, ACTOR_LR_RATE)
        self.critic = a3c.CriticNetwork(self.sess, self.a_dims, self.s_lengths, CRITIC_LR_RATE)

        self.summary_ops, self.summary_vars = a3c.build_summaries()

        self.sess.run(tf.global_initializer())
        self.writer = tf.summary.FileWriter(SUMMARY_DIR, self.sess.graph)
        self.saver = tf.train.Saver()

        if nn_model is not None:
            saver.restore(nn_model)
            logging.info('Model restored.')

        self.entropy_record = list()


    def predict(self, metrics):
        '''
        Given a set of metrics, make a prediction (set of actions).
        '''
        assert len(metrics) == len(self.s_lengths), 'Wrong number of metrics'

        # Copy the metrics to a fixed-size np.array
        state = np.zeros(self.s_dims)

        for i, metric in enumerate(metrics):
            assert self.s_lengths[i] == len(metric), \
                    f'Metric {i} has the wrong length (expected {self.s_lengths[i]}, got {len(metric)})'
            state[i, :len(metric)] = metric

        # Get, for each action, the probabilities and sample an action
        actions_prob = actor.predict(np.reshape(state, (1, self.s_dims[0], self.s_dims[1])))
        actions = list()
        for action_prob in actions_prob:
            action_cumsum = np.cumsum(action_prob)
            actions.append((action_cumsum > np.random.random()).argmax())

        self.entropy_record.append(a3c.compute_entropy(actions_prob[0]))

        return actions


    def train(self, actions, metrics, reward, epoch):
        action_vecs = [np.zeros(a_dim) for a_dim in self.a_dims]
        for i, action in enumerate(actions):
            action_vecs[i][action] = 1

        actor_gradient, critic_gradient, td_batch = None, None, None #a3c.compute_gradients(...)
        td_loss = np.mean(td_batch)

        summary_str = sess.run(summary_ops, feed_dict = {
            summary_vars[0]: td_loss,
            summary_vars[1]: reward,
            summary_vars[2]: np.mean(entropy_record)
            })

        writer.add_summary(summary_str, epoch)
        writer.flush()

        actor.apply_gradients(actor_gradient)
        critic.apply_gradient(critic_gradient)

        if epoch % MODEL_SAVE_INTERVAL == 0:
            save_path = saver.save(sess, SUMMARY_DIR, f'nn_model_ep_{epoch}.ckpt')
            logging.info(f'Model saved in file {save_path}')
