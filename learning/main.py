#!/usr/bin/env python3

import logging
import tensorflow as tf

import experiments
import stormmetrics
import learning

LOG_FILE = './agent.log'


def main(args):
    topology_name  = args['topology']
    previous_model = args['nn_model']
    max_runs       = args['max_runs']

    # Load the topology (from argument) and create the Storm runner
    topology = experiments.Topology.load(topology_name)
    runner   = stormmetrics.TopologyRunner(topology)

    # Length of each action space
    # e.g.: split_bolt_num (10 possible values), backpressure.enable (2 possible values)...
    actions        = sorted(topology.configurable_params.keys())
    actions_values = [topology.configurable_params[a] for a in actions]
    actions_space  = [len(v) for v in actions_values]

    # Length of each metric
    # e.g.: throughput (1 value), tail latency (5 values), CPU utilization (12 values)...
    metrics        = sorted(runner.metrics.keys())
    metrics_length = [runner.metrics[m] for m in metrics]

    state_length   = [len(actions)] + metrics_length


    with tf.Session() as sess, open(LOG_FILE, 'w') as log_file:

        # The learner instance communicates with the ML framework.
        learner = learning.Learner(sess, actions_space, state_length, previous_model)

        run = None
        epoch = 0

        # Training loop
        while max_runs is None or epoch < max_runs:
            logging.info('=== Epoch: %s' % epoch)
            #########
            # State #
            #########
            if run is not None:
                # Extract run metrics information (in sorted order)
                state = []
                state.append([run.config[action] for action in actions])
                for metric in metrics:
                    state.append(stormmetrics.get_metrics(run.results)[metric])
            else:
                # Fake initial metric information (don't train with those)
                state = [l * [0] for l in [len(actions)] + metrics_length]

            ##########
            # Action #
            ##########
            actions_choices = learner.predict(state)
            # Map predicted action to actual configuration parameters
            config = {}
            for action, values, choice in zip(actions, actions_values, actions_choices):
                config[action] = values[choice]

            ###########
            # Metrics #
            ###########

            run = runner.run(config)

            ##########
            # Reward #
            ##########
            reward = reward_function(stormmetrics.get_metrics(run.results))

            log_file.write('\t'.join([
                                      str(epoch),
                                      str(actions),
                                      str(reward),
                                      str(metrics),
                                      str(learner.entropy_record[-1])
                                     ]) + '\n')
            log_file.flush()


            if epoch > 0:
                learner.train(actions_choices, state, reward, epoch)

            epoch += 1


def reward_function(metrics):
    '''
    Compute the reward based on metric information about a run.
    This function defines the target QoS.
    '''
    return -metrics[latency][0]


if __name__ == '__main__':
    main({
          'topology': 'RollingCount',
          'nn_model': None,
          'max_runs': None,
         })

