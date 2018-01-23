from __future__ import print_function

import os

import yaml
import random
import pickle
from collections import defaultdict


# Possible actions (configurations)

ACTION_LABELS = (
        'topology.workers', 'topology.acker.executors',
        'topology.max.spout.pending', 'component.spout_num',
        )

ACTIONS = [
        (12, 12, 5000,  6), ( 9,  6, 7000,  6), (12,  3, 1000,  3),
        ( 6, 12, 1000,  6), (12,  3, 5000,  6), ( 9, 12, 3000,  9),
        (12,  9, 3000, 12), (12,  3, 5000,  9), (12,  3, 3000,  9),
        ( 6,  6, 1000, 12), ( 6, 12, 7000, 12), (12, 12, 7000,  9),
        ( 6,  6, 7000, 12), ( 9,  9, 5000,  9), ( 9, 12, 1000,  3),
        ( 9, 12, 7000,  3), ( 9, 12, 3000, 12), (12,  6, 3000, 12),
        ( 9,  3, 5000,  9), (12,  3, 3000,  6),
        ]

ACTIONS_NUM_SAMPLES = 6

# Jobs and performance information

JOBS = {
        'RollingCount': 'rollingcount.yaml',
        'RollingTopWords': 'rollingtopwords.yaml',
        'SentimentAnalysis': 'sentiment.yaml',
        }

PERF_LABELS = ['throughput', 'lat_50', 'lat_80', 'lat_99']

# File names

MEASUREMENTS_FILE = 'measurements.pckl'
ACTIONS_LOG_FILE = 'hidden_actions.log'

# Benchmark execution configuration

EXEC_DIR = os.getcwd()
CONFIG_DIR = EXEC_DIR + '/yamlconfs/'
STORM_CONF_DIR = os.environ['HOME'] + '/.storm/'
BENCH_DIR = os.environ['HOME'] + '/stormbenchmark/tuning/'
RESULTS_FILE = 'numbers.csv'

class Simulator:

    def __init__(self, random_seed = 42):
        """
        Init function of the performance measurements environment.
        """

        # Creates the performance measurements environment
        random.seed(random_seed)
        self.next_job = 'RollingCount' # Default job

        # Reload the history of past measurements
        try:
            with open(MEASUREMENTS_FILE) as history:
                self.measurements = pickle.load(history)
        except IOError:
            # No past measurement exists
            self.measurements = dict()
            for job in JOBS:
                self.measurements[job] = defaultdict(list)

    def get_training_measurements(self):
        """
        Sets the next hidden job and return a list of previous measurements for
        that job.
        """
        # Select a random next hidden job
        job = random.choice(JOBS.keys())
        # Choose the number of information samples returned in [0,8)
        num_samples = min(
                len(self.measurements[job]),
                random.randrange(2, ACTIONS_NUM_SAMPLES + 1)
                )
        # Select which actions to return and choose a random measurement
        available_actions = self.measurements[job].keys()
        sampled_actions = random.sample(available_actions, num_samples)

        samples = []
        for action in sampled_actions:
            measurement = random.choice(self.measurements[job][action])
            samples.append((action, measurement))

        # Record the next job secretely and return the training
        self.next_job = job
        return samples

    def get_performance(self, action_index):
        """
        Get performance of the next internally scheduled job with specified
        action. The scheduled job is hidden and changes when a new training
        input is generated.
        The action taken and resulting measurements will be saved to disk.
        """
        # Record the chosen action for each hidden topology
        with open(ACTIONS_LOG_FILE, 'a') as action_log:
            action_log.write(
                    str(action_index).rjust(2)
                    + ' '
                    + self.next_job.ljust(17)
                    + '\n'
                    )

        # If we have past measurements, optionally reuse them
        # The more measurements we have, the less we need to run the
        # configuration again
        past_measurements = self.measurements[self.next_job][action_index]
        if len(past_measurements) > 0:
            #thresh = 1 / (1 + len(past_measurements))
            thresh = 0 # this setup can't run the tests
            if random.random() >= thresh:
                return random.choice(past_measurements)

        # Run the configuration and collect results
        self.run_evaluation(ACTIONS[action_index])
        results = self.collect_last_results()

        # Record measurements and save to disk
        past_measurements.append(results)
        with open(MEASUREMENTS_FILE, 'w') as history:
            pickle.dump(self.measurements, history)

        return results

    def write_config(self, job, action):
        """
        Write the configuration corresponding to this job and action in the
        Storm config directory.
        """
        conf = yaml.load(open(CONFIG_DIR + JOBS[job]))
        for label, value in zip(ACTION_LABELS, action):
            conf[label] = value
        yaml.dump(conf, open(CONFIG_DIR + JOBS[job], 'w'), default_flow_style = False)

    def run_evaluation(self, action):
        """
        Execute the hidden topology with the chosen configuration (action).
        """
        self.write_config(self.next_job, action)
        self._set_environ()
        os.system('cp %s %s' % (CONFIG_DIR + JOBS[self.next_job], STORM_CONF_DIR))
        os.chdir(BENCH_DIR)
        os.system('bash run_nnopti.sh 1')
        os.chdir(EXEC_DIR)

    def _set_environ(self):
        os.environ['TOPOLOGY'] = self.next_job
        os.environ['CONF'] = JOBS[self.next_job]
        os.environ['STORM_HOME'] = os.environ['HOME'] + '/ansible-test/storm/apache-storm-1.0.1'
        os.environ['REDIS_HOME'] = os.environ['HOME'] + '/bilal/redis-3.2.0/src'
        os.environ['TDIGEST_JAR'] = os.environ['HOME'] + '/bilal/TDigestService/target/TDigestService-1.0-SNAPSHOT-jar-with-dependencies.jar'
        os.environ['BENCHMARK_TIME'] = '200' # should be 200 / fails below 100
        os.environ['TSERVER_PORT'] = '11111'


    def collect_last_results(self):
        """
        Collect the last results written to file by run_nnopti.sh.
        """
        os.chdir(BENCH_DIR)
        with open(RESULTS_FILE, 'r') as csv:
            lines = csv.readlines()
            last_results = lines[-1].strip()
        os.remove(RESULTS_FILE)

        print('Collected results:', last_results)
        os.chdir(EXEC_DIR)
        splits = last_results.split(',')
        throughput = int(splits[-7])
        lat_50 = float(splits[-6])
        lat_80 = float(splits[-3])
        lat_99 = float(splits[-1])
        return (throughput, lat_50, lat_80, lat_99)
