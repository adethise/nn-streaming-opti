#!/usr/bin/env python3

import experiments
from datetime import datetime


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

        results = {
                'throughput': [12100],
                'tail_latencies': [20, 40, 75, 120, 190]
                }

        self.runs.append(experiments.Run(self.topology.name, timestamp, config, results))

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
