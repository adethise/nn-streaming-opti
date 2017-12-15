import yaml
import random
import os

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

JOBS = {
        'RollingCount': 'rollingcount.yaml',
        'RollingTopWords': 'rollingtopwords.yaml',
        'SentimentAnalysis': 'sentiment.yaml',
        }

EXEC_DIR = os.getcwd()
CONFIG_DIR = os.getcwd() + '/yamlconfs/'
STORM_CONF_DIR = '/home/ubuntu/.storm/'
BENCH_DIR = '/home/ubuntu/stormbenchmark/tuning/'

class Simulator:

    def __init__(self, random_seed = 42):
        random.seed(random_seed)
        self.next_job = 'RollingCount'

    def get_performance(self, action_index):
        self.run_evaluation(ACTIONS[action_index])

    def write_config(self, job, action):
        conf = yaml.load(open(CONFIG_DIR + JOBS[job]))
        for label, value in zip(ACTION_LABELS, action):
            conf[label] = value
        yaml.dump(conf, open(CONFIG_DIR + JOBS[job], 'w'), default_flow_style = False)

    def run_evaluation(self, action):
        self.write_config(self.next_job, action)
        os.system('cp %s %s' % (CONFIG_DIR + JOBS[self.next_job], STORM_CONF_DIR))
        os.chdir(BENCH_DIR)
        os.system('bash run_nnopti.sh 1')
        os.chdir(EXEC_DIR)

    def _set_environ(self):
        os.environ['TOPOLOGY'] = self.next_job
        os.environ['CONF'] = JOBS[self.next_job]
        os.environ['STORM_HOME'] = '~/ansible-test/storm/apache-storm-1.0.1'
        os.environ['REDIS_HOME'] = '~/bilal/redis-3.2.0/src'
        os.environ['TDIGEST_JAR'] = '~/bilal/TDigestService/target/TDigestService-1.0-SNAPSHOT-jar-with-dependencies.jar'
        os.environ['BENCHMARK_TIME'] = '30' # should be 200
        os.environ['TSERVER_PORT'] = '11111'


    def collect_last_results(self):
        os.chdir(BENCH_DIR)
        with open('results.csv', 'r') as csv:
            lines = csv.readlines()
            last_results = lines[-1]

        print('Result line:', last_results)
        return last_results
