#!/usr/bin/env python3

import os
import json

TOPO_DIR = 'topologies'
RUNS_DIR = 'runs'

class Topology:
    '''
    Define topologies and stores information required to run them.
    '''

    def __init__(self, name, classpath, default_params, configurable_params):
        '''
        Create a new topology.

        ::params::
        name - the name of this topology for reference and location
        classpath - the java classpath of the topology to send to Storm
        default_params - a set of default parameters to run the topology
            this is a dict() of param: value
        configurable_params - a set of parameters than can override the default values
            this is a dict() of param: [value1, value2,...]
        '''

        self.name = name
        self.classpath = classpath
        self.default_params = default_params
        self.configurable_params = configurable_params

    def save(self, topo_dir = TOPO_DIR):
        '''
        Save this topology under `topo_dir`.
        Note that topologies can be created manually - this is an helper function
        for automatic topology creation.
        '''
        try:
            os.mkdir(topo_dir)
        except FileExistsError:
            pass

        with open(os.path.join(topo_dir, self.name), 'x') as _file:
            json.dump({
                'name': self.name,
                'classpath': self.classpath,
                'default_params': self.default_params,
                'configurable_params': self.configurable_params,
                }, _file, indent = 4)

    @staticmethod
    def load(name, topo_dir = TOPO_DIR):
        '''
        Load a topology and related information.
        '''
        with open(os.path.join(topo_dir, name)) as _file:
            data = json.load(_file)

            name = data['name']
            classpath = data['classpath']
            default_params = data['default_params']
            configurable_params = data['configurable_params']

        return Topology(name, classpath, default_params, configurable_params)


class Run:
    '''
    Class to hold results for each experiment run.
    '''

    def __init__(self, topology, timestamp, config, results, run_name = None):
        '''
        Create a new run record.

        ::params::
        topology - the name of the topology
        timestamp - identify the experiment in time
        config - parameters that were set (non-default) for this run
        results - measurements
        run_name - the name of the folder where this run should be recorded (optional)
        '''

        self.topology = topology
        self.timestamp = timestamp
        self.config = config
        self.results = results

        # The default run name is <topology>/<timestamp> and
        # will create the run in a subfolder of <topology>
        if run_name:
            self.run_name = run_name
        else:
            self.run_name = os.path.join(self.topology, self.timestamp)


    def save(self, runs_dir = RUNS_DIR):
        '''
        Save this run under `runs_dir`.
        '''
        savedir = os.path.join(runs_dir, self.run_name)
        os.makedirs(savedir) # Returns an error if the folder already exists

        # Multiple types of information will be saved in separate files
        # for simplicity and human-readability

        # Save the metadata
        # This should include collected information such as errors or system status
        with open(os.path.join(savedir, 'metadata'), 'x') as _file:
            json.dump({'topology': self.topology, 'timestamp': self.timestamp}, _file, indent = 4)

        # Save the configuration for this run
        with open(os.path.join(savedir, 'config'), 'x') as _file:
            json.dump(self.config, _file, indent = 4)

        # Save the measurements for this run
        with open(os.path.join(savedir, 'results'), 'x') as _file:
            json.dump(self.results, _file, indent = 4)

    @staticmethod
    def load(run_name, runs_dir = RUNS_DIR):
        '''
        Loads and returns a previous run. Complement of save().
        '''
        savedir = os.path.join(runs_dir, run_name)

        with open(os.path.join(savedir, 'metadata')) as _file:
            metadata = json.load(_file)
            topology = metadata['topology']
            timestamp = metadata['timestamp']

        with open(os.path.join(savedir, 'config')) as _file:
            config = json.load(_file)

        with open(os.path.join(savedir, 'results')) as _file:
            results = json.load(_file)

        return Run(topology, timestamp, config, results, run_name = run_name)

