import os
timeout = None


def run_task_sleep(task_info, taskdir, tempdir):
    import numpy as np
    import time
    run_time = task_info['sleep']
    print('Sleep for {}. taskdir:{} tempdir:{}'.format(run_time, taskdir, tempdir))
    time.sleep(run_time)  # Do work (but first a nap!)
    asum = task_info['array'].sum()
    results = {'raw_data':  {'seed':  task_info['seed']},
               'sim_state': {'sleep': task_info['sleep']},
               'computed':  {'sum':   np.array([asum])}}
    return results


class JobInfo(object):
    run_task = staticmethod(run_task_sleep)

    def __init__(self, root_dir, job_dir):
        import time
        import sys
        self.job_dir = job_dir
        if self.job_dir == '':
            self.job_dir = os.path.join(root_dir, 
                                        os.path.basename(sys.argv[0])[:-3],
                                        time.strftime('%Y-%m-%d-%Hh%Mm%Ss'))
        self.job_dir = os.path.expanduser(self.job_dir)
        self.n_tasks = 50
        self.maxsleep = 3
        self._job_info = None
        self._task_names = None

    def prepare_tasks(self):
        import random
        import numpy as np

        if not os.path.exists(self.job_dir):
            os.mkdir(self.job_dir)

        self._job_info = {i+100: {'sleep': random.randint(1, self.maxsleep),
                                  'array': np.random.rand(1000,1000),
                                  'seed':  0} for i in range(self.n_tasks)}
        self._task_names = {task_id: 'task{0}'.format(task_id)
                            for task_id in self._job_info.keys()}

        return self._job_info.keys()

    def get_task_info(self, task_id):
        return self._job_info[task_id] 

    def task_name(self, task_id):
        return 'task{0}'.format(task_id)

    def record_result(self, task_id, simresult, taskdir):
        import six.moves.cPickle as cPickle
        from snep.parallel2 import SimulationResult
        assert isinstance(simresult, SimulationResult)
        os.mkdir(taskdir)  # if you want to put something in the taskdir, you must create it first
        fn = os.path.join(taskdir, 'results.pickle')
        with open(fn, 'wb') as f:
            cPickle.dump(simresult, f, protocol=cPickle.HIGHEST_PROTOCOL)
        print(simresult.task_id, simresult.status, simresult.finaldata)
    
    def log_info(self, msg):
        print(msg)
    
    def log_err(self, msg):
        print(msg)
    
    def finalize(self, delete_prev):
        print('Finalizing job.')

    def resume_state(self, statepath):
        pass

    @staticmethod
    def extract_state(job_name, resultdir):
        print("Please choose a previous job to resume from: ")
        return None

if __name__ == '__main__':
    from snep.parallel2 import run
    '''
    IMPORTANT: Only include code here that can be run repeatedly,
    because this will be run once in the parent process, and then
    once for every worker process.
    '''
    ji_kwargs = dict(root_dir=os.path.expanduser('~/experiments'))
    job_info = run(JobInfo, ji_kwargs, max_task_time=timeout)
