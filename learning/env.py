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

CONFIG_DIR = 'yamlconfs/'

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
        os.environ['TOPOLOGY'] = self.next_job
        os.environ['CONF'] = JOBS[self.next_job]
        os.system('cp %s ~/.storm/' % (CONFIG_DIR + JOBS[self.next_job]))
        os.chdir('/home/ubuntu/stormbenchmark/tuning')
        os.system('bash run_nnopti.sh 1')
