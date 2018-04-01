#!/usr/bin/env python3

import logging
import os
import tensorflow as tf
import numpy as np

import a3c

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001

SUMMARY_DIR = './learning_results'
MODEL_SAVE_INTERVAL = 10


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

        self.sess.run(tf.global_variables_initializer())
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
        state = self._metrics_to_array(metrics)

        # Get, for each action, the probabilities and sample an action
        actions_prob = self.actor.predict(np.reshape(state, (1, self.s_dims[0], self.s_dims[1])))
        actions = list()
        for action_prob in actions_prob:
            action_cumsum = np.cumsum(action_prob)
            actions.append((action_cumsum > np.random.random()).argmax())

        entropy = a3c.compute_entropy(np.concatenate([a[0] for a in actions_prob]))
        self.entropy_record.append(entropy)

        return actions


    def train(self, actions, metrics, reward, epoch):
        # Convert actions choices to full certainty prediction
        action_vecs = tuple(np.zeros((1, a_dim)) for a_dim in self.a_dims)
        for i, action in enumerate(actions):
            action_vecs[i][0, action] = 1

        # Copy the metrics to a fixed-size np.array
        state = self._metrics_to_array(metrics)

        # Compute the update gradients
        actor_gradient, critic_gradient, td_batch = a3c.compute_gradients([state], [action_vecs], [reward],
                terminal = False, actor = self.actor, critic = self.critic)

        td_loss = np.mean(td_batch)

        summary_str = self.sess.run(self.summary_ops, feed_dict = {
            self.summary_vars[0]: td_loss,
            self.summary_vars[1]: reward,
            self.summary_vars[2]: np.mean(self.entropy_record)
            })

        self.writer.add_summary(summary_str, epoch)
        self.writer.flush()

        self.actor.apply_gradients(actor_gradient)
        self.critic.apply_gradients(critic_gradient)

        if epoch % MODEL_SAVE_INTERVAL == 0:
            save_path = self.saver.save(self.sess, os.path.join(SUMMARY_DIR, 'nn_model_ep_%s.ckpt' % epoch))
            logging.info('Model saved in file %s' % save_path)


    def _metrics_to_array(self, metrics):
        state = np.zeros(self.s_dims)

        for i, metric in enumerate(metrics):
            assert self.s_lengths[i] == len(metric), \
                    'Metric %s has the wrong length (expected %s, got %s)' % (i, self.s_lengths[i], len(metric))
            state[i, :len(metric)] = metric

        return state
