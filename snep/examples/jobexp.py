from snep.configuration import config
from snep.experiment import Experiment
import os

config['cluster'] = config.run_on_cluster()

username = None
max_tasks = 32
mem_per_task = 1  # in GB
max_task_time = '00:10:00'  # In HH:MM:SS
poll_interval = 1.  # in minutes


def run_task_sleep(task_info, taskdir, tempdir):
    import time
    import numpy as np
    os.mkdir(taskdir)  # if you want to put something in the taskdir, you must create it first
    run_time = task_info['sleep']
    print('sleep for {}. taskdir:{} tempdir:{}'.format(run_time, taskdir, tempdir))
    time.sleep(run_time)  # Do work (but first a nap!)
    os.rmdir(taskdir)
    asum = task_info['y']['b'].sum()
    results = {'raw_data': task_info['y'],
               'sim_state': {'sleep': np.array([task_info['sleep']])},
               'computed': {'sum': np.array([asum])}}
    return results


class JobInfoExperiment(Experiment):
    run_task = staticmethod(run_task_sleep)

    def __init__(self, *args, **kwargs):
        """
        This constructor should contain code used to initialize or modify the job, regardless of whether
         snep.parallel.run is True (in which case _update_tasks is called) or False (_prepare_tasks is called).
        :param args:
        :param kwargs:
        :return:
        """
        super(JobInfoExperiment, self).__init__(*args, **kwargs)

    def _update_tasks(self):
        """
        This function is called when the 'resume' parameter of snep.parallel.run is True.
        If you are resuming a job, the Experiment class will automatically copy all the parameters
        and parameter ranges from the resumed h5f file. It may be desirable to modify the run time
        of the simulation, or other non-parameterspace parameters.
        :return: None
        """
        pass

    def _prepare_tasks(self):
        from snep.utils import Parameter, ParameterArray, ParametersNamed
        import numpy as np

        param_fixed = {'x': {'r': Parameter(50., 'volt'),
                             },
                       'y': {'a': Parameter(-12),
                             },
                       }
        self.tables.add_parameters(param_fixed)

        param_ranges = {'x': {'q': {'gl': ParameterArray(np.arange(9., 13.), 'nsiemens')
                                    }
                              },
                        'y': {'b': ParametersNamed([(str(b), ParameterArray(np.random.rand(1000, 1000)))
                                                    for b in range(3)])
                              },
                        'sleep': ParametersNamed([('{:.1f}'.format(s), s) for s in np.linspace(1., 4., 3)])
                        }
        self.tables.add_parameter_ranges(param_ranges)

        self.tables.link_parameter_ranges([('y', 'b'), ('sleep',)])


if __name__ == '__main__':
    from snep.parallel2 import run

    '''
    IMPORTANT: Only include code here that can be run repeatedly,
    because this will be run once in the parent process, and then
    once for every worker process.
    '''
    ji_kwargs = dict(root_dir=os.path.expanduser('~/experiments'))
    job_info = run(JobInfoExperiment, ji_kwargs, username=username, max_tasks=max_tasks, mem_per_task=mem_per_task,
                   max_task_time=max_task_time, poll_interval=poll_interval)
