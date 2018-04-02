#!/usr/bin/env python3

import experiments

WORKERS = 5

rc_name = 'RollingCount'

rc_classpath = 'storm.benchmark.benchmarks.RollingCount'

rc_default_params = {
        'benchmark.spout.tuple_limit': 450000,
        'component.rolling_count_bolt_num': 3 * WORKERS,
        'component.split_bolt_num': 3 * WORKERS,
        'component.spout_num': 2 * WORKERS,
        'emit.frequency': 1,
        'benchmark.spout.interval': 10000,
        'metrics.enabled': True,
        'metrics.path': 'reports',
        'metrics.poll': 10000,
        'metrics.time': 3600000,
        'spout.file': '/A_Tale_of_Two_City.txt',
        'topology.acker.executors': 2 * WORKERS,
        'topology.executor.receive.buffer.size': 8192,
        'topology.executor.send.buffer.size': 8192,
        'topology.max.spout.pending': 4000,
        'topology.name': 'RollingCount',
        'topology.tdigestserver': '130.104.230.106',
        'topology.transfer.buffer.size': 8192,
        'topology.tdigestserver.port': 11111,
        'topology.worker.childopts': '-Xmx8g -Djava.net.preferIPv4Stack=true',
        'topology.workers': 2 * WORKERS,
        'window.length': 10,
        'worker.profiler.enabled': True,
        }

rc_conf_params = {
        'topology.workers': list(range(2 * WORKERS, 4 * WORKERS + 1, WORKERS)),
        'component.rolling_count_bolt_num': list(range(WORKERS, 20 * WORKERS + 1, WORKERS)),
        'component.split_bolt_num': list(range(WORKERS, 20 * WORKERS + 1, WORKERS)),
        'component.spout_num': list(range(WORKERS, 12 * WORKERS + 1, WORKERS)),
        'topology.acker.executors': list(range(WORKERS, 12 * WORKERS + 1, WORKERS)),
        'topology.max.spout.pending': list(range(1000, 10001, 2000)),
        'topology.worker.receiver.thread.count': list(range(1, 4, 1)),
        'topology.backpressure.enable': [False, True],
        'topology.worker.shared.thread.pool.size': list(range(4, 9, 2)),
        'topology.bolts.outgoing.overflow.buffer.enable': [False, True],
        'topology.disruptor.batch.size': list(range(100, 401, 100)),
        'topology.disruptor.batch.timeout.millis': list(range(1, 4, 1)),
        }

rollingcount = experiments.Topology(rc_name, rc_classpath, rc_default_params, rc_conf_params)

if __name__ == '__main__':
    rollingcount.save()
