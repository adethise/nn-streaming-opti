from __future__ import print_function

import os

import yaml
import random
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

# Jobs and performance information

JOBS = {
        'RollingCount': 'rollingcount.yaml',
        'RollingTopWords': 'rollingtopwords.yaml',
        'SentimentAnalysis': 'sentiment.yaml',
        }

PERF_LABELS = ['throughput', 'lat_50', 'lat_80', 'lat_99']


# Benchmark execution configuration

EXEC_DIR = os.getcwd()
CONFIG_DIR = os.getcwd() + '/yamlconfs/'
STORM_CONF_DIR = os.environ['HOME'] + '/.storm/'
BENCH_DIR = os.environ['HOME'] + '/stormbenchmark/tuning/'

class Simulator:

    def __init__(self, random_seed = 42):
        random.seed(random_seed)
        self.next_job = 'RollingCount'

        self.measurements = dict()
        for job in JOBS:
            self.measurements[job] = defaultdict(list)

    def get_performance(self, action_index):
        self.run_evaluation(ACTIONS[action_index])
        results = self.collect_last_results()
        self.measurements[self.next_job][action_index].append(results)

    def write_config(self, job, action):
        conf = yaml.load(open(CONFIG_DIR + JOBS[job]))
        for label, value in zip(ACTION_LABELS, action):
            conf[label] = value
        yaml.dump(conf, open(CONFIG_DIR + JOBS[job], 'w'), default_flow_style = False)

    def run_evaluation(self, action):
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
        os.environ['TDIGEST_JAR'] = os.environ['HOME'] + \
                '/bilal/TDigestService/target/TDigestService-1.0-SNAPSHOT-jar-with-dependencies.jar'
        os.environ['BENCHMARK_TIME'] = '200' # should be 200 / fails below 100
        os.environ['TSERVER_PORT'] = '11111'


    def collect_last_results(self):
        os.chdir(BENCH_DIR)
        with open('numbers.csv', 'r') as csv:
            lines = csv.readlines()
            last_results = lines[-1]

        print('Collected results:', last_results)
        os.chdir(EXEC_DIR)
        splits = last_results.split(',')
        throughput = splits[-7]
        lat_50 = splits[-6]
        lat_80 = splits[-3]
        lat_99 = splits[-1]
        return (throughput, lat_50, lat_80, lat_99)
