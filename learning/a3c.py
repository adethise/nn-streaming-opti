#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tflearn


GAMMA = 0.99
ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6

NETWORK_DEPTH = 3

'''
Implements the A3C algorithm and the neural network for the performance
optimization problem.

The input of the model is the following:
    a list of vectors, each of length :s_lengths[i]:, stored in a 3D array

    **Example**: s_lengths = [1, 3, 4]
    metrics: [[1200],
              [48, 99, 120],
              [0.1, 0.3, 0.2, 0.4]]

    inputs: [[[1200,    0,    0,    0],
              [  48,   99,  120,    0],
              [ 0.1,  0.3,  0.2,  0.4]]]

    N.B.: the value passed to predict() is the :inputs: array.

The output of the model is the following:
    a list of vectors, each of length :a_dims[i]:

    **Example**: a_dims = [2, 5, 5]
    outputs: [[0, 1],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0]]
'''


def create_network(s_lengths):
    '''
    Create the neural network.

    :a_dim: array containing the dimension space for each action
    :s_len: array containing the length of each metric information
    :last_layer: activation mode for the output neurons
    '''
    inputs = tflearn.input_data(shape=[None, len(s_lengths), max(s_lengths)])
    splits = list()

    # Add a convolution layer for each input vector
    for i, s_len in enumerate(s_lengths):
        splits.append(tflearn.conv_1d(inputs[:, i:i+1, :s_len], 128, 4, activation = 'relu', name = 'Input%s' % i))

    # Merge all initial convolution layers
    dense_net = tflearn.merge(splits, 'concat', name = 'MergeNet')

    # Hidden layers
    for i in range(NETWORK_DEPTH):
        dense_net = tflearn.conv_1d(dense_net, 128, 4, activation = 'relu', name = 'Dense%s' % i)

    return inputs, dense_net



class ActorNetwork(object):
    """
    Input to the network is the state, output is the distribution
    of all actions.
    """
    def __init__(self, sess, a_dims, s_lengths, learning_rate):
        self.sess = sess
        self.a_dims = a_dims
        self.s_lengths = s_lengths
        self.lr_rate = learning_rate

        # Create the actor network
        self.inputs, self.outputs = self.create_actor_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape(), name = 'NetworkParam'))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected actions, tuple of 0-1 vectors
        self.acts = tuple(tf.placeholder(tf.float32, [None, a_dim], 'PlaceholderAction%s' % i) \
                for i, a_dim in enumerate(self.a_dims))

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1], 'GradWeights')

        # Compute the objective (log action_vector and entropy) # Dimensions
        actions_value = sum(tf.reduce_sum(out * act, axis = 1)
                for out, act in zip(self.outputs, self.acts))
        actions_obj = tf.reduce_sum(tf.log(actions_value) * -self.act_grad_weights)

        entropy_value = sum(tf.reduce_sum(out * tf.log(out + ENTROPY_EPS))
                for out in self.outputs)
        entropy_obj = tf.reduce_sum(entropy_value)

        self.obj = actions_obj + ENTROPY_WEIGHT * entropy_obj

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

    def create_actor_network(self):
        with tf.variable_scope('actor'):
            inputs, dense_net = create_network(self.s_lengths)

            # Add one relu and one softmax output layer for each type of action
            # The layers have as many neurons as possible choices for each action
            outputs = list()
            for i, a_dim in enumerate(self.a_dims):
                out = tflearn.fully_connected(dense_net, 128, activation = 'relu', name = 'ReluAction%s' % i)
                outputs.append(tflearn.fully_connected(out, a_dim, activation = 'softmax', name = 'OutputAction%s' % i))

            # A tuple of tensors is a valid input for most tflearn functions
            return inputs, tuple(outputs)

    def train(self, inputs, acts, act_grad_weights):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={
            self.inputs: inputs
        })

    def get_gradients(self, inputs, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.inputs: inputs,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, a_dims, s_lengths, learning_rate):
        self.sess = sess
        self.a_dims = a_dims
        self.s_lengths = s_lengths
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape(), name = 'NetworkParam'))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1], 'Target')

        # Temporal Difference, will also be weights for actor_gradients
        self.td = self.td_target - self.out

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            inputs, dense_net = create_network(self.s_lengths)

            output = tflearn.fully_connected(dense_net, 1, activation = 'linear', name = 'Output')

            return inputs, output

    def train(self, inputs, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def get_td(self, inputs, td_target):
        return self.sess.run(self.td, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def get_gradients(self, inputs, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.inputs: inputs,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(state, actions, reward, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    reward = np.reshape(reward, (1,1))
    values = reward - critic.predict(state)

    actor_gradients = actor.get_gradients(state, actions, values)
    critic_gradients = critic.get_gradients(state, reward)

    return actor_gradients, critic_gradients, values


#def discount(x, gamma):
#    """
#    Given vector x, computes a vector y such that
#    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
#    """
#    out = np.zeros(len(x))
#    out[-1] = x[-1]
#    for i in reversed(range(len(x)-1)):
#        out[i] = x[i] + gamma*out[i+1]
#    assert x.ndim >= 1
#    # More efficient version:
#    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
#    return out


def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
