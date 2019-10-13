import multiprocessing as mp
import sys
import argparse
import platform
import re

cluster_url = 'cluster.ml.tu-berlin.de'
cluster_head = 'cluster'
hostname = platform.node()
on_cluster = re.match('cognition[0-9]+', hostname) is not None
on_head = hostname == cluster_head


class Config(object):
    """
    Stores all the run time configuration options. There should only
    be one instance of this object, 'config' which should be imported
    and used like a dictionary:

    from Config import config
    config['simulationgroup_name'] = 'my_sim_group_name'
    if config['mkl']:
        etc...
    """

    def __init__(self):
        self._config = {'type': 'none',
                        'debug': sys.gettrace() is not None,
                        'image_suffix': '.svg',
                        'exclude_nodes': []}
        self.initialize_argparse()

    def initialize_argparse(self):
        from os import environ
        parser = argparse.ArgumentParser(description=
                                         'SNEP - Simultaneous Numerical Exploration Package')
        parser.add_argument('--procs', action='store', type=int,
                            default=(environ['NUMPROCS'] if 'NUMPROCS' in environ else 0),
                            help='number of processors to use (default: prompt when run)')
        parser.add_argument('--rt', action='store', type=float, default=(environ['RT'] if 'RT' in environ else 0.),
                            help='simulation run time (default: 1)')
        parser.add_argument('--jobdir', action='store', type=str,
                            default=(environ['SNEPJOBDIR'] if 'SNEPJOBDIR' in environ else ''),
                            help='subdirectory in which to store results & output (default: date-time)')
        # parser.add_argument('--mkl', action='store', type=int, default=1,
        #                     help='number of cores per process for IntelMKL (default: 1)')
        parser.add_argument('--cluster', action='store', type=bool, default=False,  # self._run_on_cluster(),
                            help='parallelize using MPI (default: true if running on the cluster)')
        # parser.add_argument('--plot', action='store_true', help='do plotting (default: False)')

        mode_str = 'new (new simulation) / resume (load state from previous) / finish (complete unfinished) / plot'
        parser.add_argument('--mode', action='store', type=str, default='new',
                            help=mode_str)

        parser.add_argument('--worker', action='store_true', help='internal argument used by MPI code')

        ns, unknown = parser.parse_known_args()
        for y in vars(ns):
            self._config[y] = getattr(ns, y)

            # if (self._config['procs'] < 1 and not self._config['debug']) \
            #         or (self._config['cluster'] and (not on_cluster or on_head)):
            #     self._config['procs'] = self.prompt_procs()

    def __getitem__(self, key):
        try:
            ret = self._config[key]
            if key == 'debug':
                print('Multiprocessing ' + ('disabled' if ret else 'enabled'))
        except KeyError:
            raise KeyError('Configuration item does not exist')
        return ret

    def __setitem__(self, key, value):
        if key not in self._config:
            raise KeyError('Configuration item does not exist')
        else:
            self._config[key] = value

    def prompt_procs(self):
        import six
        default = 1
        if not self._config['cluster']:
            default = mp.cpu_count()
        prompt = 'Enter number of worker processes ({0}): '.format(default)
        inp = 0
        while inp < 1:
            try:
                raw = six.moves.input(prompt)
                inp = default if raw == '' else int(raw)
            except ValueError:
                print('Invalid selection')
        return inp

    @staticmethod
    def run_on_cluster():
        import six
        ret = on_cluster or on_head
        if not ret:
            ret = 'y' == six.moves.input('Do you want to run on the cluster? y/[n] ')
            print('Job will be run on the cluster.' if ret else 'Job will be run locally.')
        return ret


config = Config()
