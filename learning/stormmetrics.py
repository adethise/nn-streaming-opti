#!/usr/bin/env python3

from datetime import datetime
import logging
import os
import yaml
import subprocess
import experiments


STORM_PATH = os.path.join(os.environ['HOME'], 'apache-storm/bin/storm')
BENCH_PATH = os.path.join(os.environ['HOME'], 'stormbenchmark/bin/stormbench')
JAR_FILE = os.path.join(os.environ['HOME'], 'stormbenchmark/target/storm-benchmark-0.1.0-jar-with-dependencies.jar')

RUN_TIME_SECONDS = 200

METRICS = { # Lenght of each metrics information
        'throughput': 1,
        'tail_latencies': 5, # lat_50, lat_80, lat_90, lat_95, lat_99
        }


class TopologyRunner:
    def __init__(self, topology):
        self.topology = topology
        self.runs = list()
        self.metrics = METRICS

    def run(self, params, save = True):
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M')
        config = self.topology.default_params.copy()
        config.update(params)

        logging.info('Creating config file...')
        config_file = os.path.join(os.getcwd(), timestamp + '.yaml')
        with open(config_file, 'w') as _file:
            yaml.dump(config, _file)

        logging.info('Running the benchmark...')
        bench = subprocess.Popen([
                BENCH_PATH,
                '-storm', STORM_PATH,
                '-jar', JAR_FILE,
                '-conf', config_file,
                'storm.benchmark.tools.Runner', self.topology.classpath
                ], universal_newlines = True)

        try:
            bench.wait(RUN_TIME_SECONDS)
        except subprocess.TimeoutExpired:
            bench.terminate()

        results = {'returncode': [bench.returncode]}
        os.unlink(config_file)

        logging.info('Saving the results...')
        self.runs.append(experiments.Run(self.topology.name, timestamp, config, results))
        if save:
            self.runs[-1].save()

        logging.info('Killing the topology...')
        subprocess.run([STORM_PATH, 'kill', self.topology.name, '-w', '1'])

        return self.runs[-1]


def _random_config(params):
    '''
    Create a random configuration from a configurable_params dict.
    This can be used to generate random samples.
    '''
    import random
    return {k: random.choice(v) for k, v in params.items()}


if __name__ == '__main__':
    rc = experiments.Topology.load('RollingCount')
    runner = TopologyRunner(rc)

    runner.run(_random_config(rc.configurable_params))