import logging
import os
import sys
import threading
import time

from snep.tables.experiment import ExperimentTables


class Experiment(object):
    def __init__(self,
                 root_dir,
                 suffix=None,
                 flat_results_groups=True,
                 run_only_unfinished=True,
                 job_dir='',
                 ):
        """
                 repeat_experiment_dir=None,
                 repeat_experiment_only_unfinished=False,
                 repeat_experiment_new_paramspace=False,
        repeat_experiment_dir - To repeat an experiment, pass the directory it is contained in
                    for this parameter, and the network tables will be copied into a new
                    tables file and the parameter space will be re-simulated.
        repeat_experiment_only_unfinished - Resumes the experiment at repeat_experiment_dir
                    writing the results from any previously unfinished simulations
                    into the existing file.
        repeat_experiment_new_paramspace - Copies experiment from repeat_experiment_dir
                    but removes all the parameter ranges, effectively deleting
                    the original parameter space.
        flat_results_groups - if true the results are stored in directories and groups that are only
                one deep, otherwise they are stored in a nested hierarchy.
        """

        self.root_dir = os.path.expanduser(root_dir)

        self.run_only_unfinished = run_only_unfinished
        self.prev_tables = None
        if job_dir == '':
            # if repeat_experiment_only_unfinished:
            #     result_dir = repeat_experiment_dir
            # else:
            timestr = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
            result_dir = '-'.join((timestr, suffix)) if suffix else timestr

            experiment_dir = os.path.join(self.root_dir,
                                          os.path.basename(sys.argv[0])[:-3],
                                          result_dir)

            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
        else:
            experiment_dir = job_dir

        self.experiment_dir = os.path.expanduser(experiment_dir)

        self.tables = ExperimentTables(os.path.join(self.experiment_dir, 'experiment.h5'),
                                       flat_results_groups)
        self.tables.open_file()
        self.tables.initialize()

        # if not repeat_experiment_only_unfinished and repeat_experiment_dir:
        #     repeat_tables_path = os.path.join(self.root_dir,repeat_experiment_dir,'experiment.h5')
        #     repeat_tables = ExperimentTables(repeat_tables_path)
        #     repeat_tables.open_file(readonly=True)
        #     repeat_tables.initialize()
        #     repeat_tables.copy_network(self.tables)
        #     if repeat_experiment_new_paramspace:
        #         self.tables.delete_all_parameter_ranges()
        #     else:
        #         repeat_tables.copy_paramspace(self.tables)
        #     repeat_tables.close_file()

        consoleformatter = logging.Formatter('%(process)d %(levelname)s %(message)s')
        stdout = logging.StreamHandler(sys.stdout)  # stdout)
        stdout.setFormatter(consoleformatter)
        stdout.setLevel(logging.INFO)
        logger = logging.getLogger('snep.experiment')
        logger.setLevel(logging.INFO)
        rl = logging.getLogger('')
        rl.addHandler(stdout)

        self.monitors = {'spikes': set(), 'statevar': {}, 'poprate': set()}

    def _set_experiment_dir(self, job_dir):
        self.experiment_dir = job_dir

    job_dir = property(fset=_set_experiment_dir, fget=lambda self: self.experiment_dir)

    def prepare_tasks(self):
        if self.prev_tables:
            self.log_info('Updating tasks from existing experiment')
            self._update_tasks()
        else:
            self.log_info('Preparing tasks for new experiment')
            self._prepare_tasks()
        return self.tables.get_task_ids(self.run_only_unfinished)

    def _update_tasks(self):
        """
        To be overridden in the derived class, allowing the user to modify any part of
        the previous experiment which is to be resumed / repeated.
        :return: None
        """
        return

    def get_task_info(self, task_id):
        from snep.utils import Parameter
        msg = 'Getting info for pid: {0}, thr: {1}, {2}'
        self.log_info(msg.format(os.getpid(),
                                 threading.current_thread().name,
                                 self.tables.get_results_group_path(task_id)))

        params = self.tables.get_task_info(task_id)
        params['monitors'] = self.monitors
        if self.prev_tables:
            if ('seed',) in task_id and self.tables.seed_map:
                task_id = dict(task_id)
                rsm = {v: k for k, v in self.tables.seed_map.items()}
                task_id[('seed',)] = Parameter(rsm[task_id[('seed',)].value])
            params['resume_state'] = self.prev_tables.get_sim_state(task_id)
        return params

    def task_name(self, task_id):
        return self.tables.task_name(task_id)

    def record_result(self, task_id, simresult, taskdir):
        import traceback
        start = time.time()
        status = 'error'
        try:
            finaldata = simresult.finaldata
            # task_id = simresult.task_id
            sim_state = finaldata.pop('sim_state', None)
            if sim_state:
                self.tables.add_sim_state(task_id, sim_state)
            raw_data = finaldata.pop('raw_data', None)
            if raw_data:
                # self.record_brian_result(task_id, raw_data)
                self.tables.add_raw_data(task_id, raw_data)
            computed = finaldata.pop('computed', None)
            if computed:
                self.tables.add_computed(task_id, computed)
            log_file = finaldata.pop('log_file', {})
            if log_file:
                self.tables.add_log_file(task_id, log_file)
                # callstack = finaldata.pop('callstack', None)
                # if callstack:
                #     self.tables.add_log_file(task_id, 'callstack', callstack)
                # exc_info = finaldata.pop('exc_info', None)
                # if exc_info:
                #     self.tables.add_log_file(task_id, 'exc_info', exc_info)
        except:
            traceback.print_exc()
            err = 'Exception occurred while collecting results for {0}'
            ExperimentTables.log_err(self.tables.h5f, err.format(simresult.task_id))
        else:
            status = simresult.status
        finally:
            # print(status, simresult.tasktime, simresult.cluster_info)
            self.tables.set_results_status(task_id, status, simresult.tasktime, simresult.cluster_info)

        total = time.time() - start
        self.log_info("Collected result file in {0:.1f} seconds.".format(total))

        # def record_brian_result(self, task_id, raw_data):

    # # Brian specific results
    #     results_group_str = self.tables.get_results_group_path(task_id)
    #     poprate = raw_data.pop('brian-poprate', None)
    #     if poprate:
    #         for pop_name, (times, rates) in poprate.iteritems():
    #             self.tables.add_population_rates(task_id, pop_name, times, rates)
    #     spikes = raw_data.pop('brian-spikes', None)
    #     if spikes:
    #         spikes_group_str = '/'.join((results_group_str, 'spikes'))
    #         self.tables.add_raw_data(task_id, raw_data)
    #         for pop_name, spikes in spikes.iteritems():
    #             self.tables.add_spiketimes(task_id, pop_name, spikes)
    #     statevar = raw_data.pop('brian-statevar', None)
    #     if statevar:
    #         for pop_name, all_vars in statevar.iteritems():
    #             for varname, (times, values) in all_vars.iteritems():
    #                 self.tables.add_state_variables(task_id, pop_name, varname, times, values)

    def add_monitors_state_variables(self, monitors):
        self.monitors['statevar'].update(monitors)

    def add_monitors_population_rate(self, monitors):
        self.monitors['poprate'].update(monitors)

    def add_monitors_spike_times(self, monitors):
        self.monitors['spikes'].update(monitors)

    def log_info(self, msg):
        logging.getLogger('snep.experiment').info('::MAINPROC:: ' + msg)
        ExperimentTables.log_info(msg, self.tables.h5f)

    def log_err(self, msg):
        logging.getLogger('snep.experiment').error('::MAINPROC:: ' + msg)
        ExperimentTables.log_err(msg, self.tables.h5f)

    def finalize(self, delete_prev):
        import shutil
        self.tables.close_file()
        if self.prev_tables:
            self.prev_tables.close_file()
            if delete_prev:
                fn = self.prev_tables.filename
                shutil.rmtree(os.path.dirname(fn))

    def resume_state(self, statepath):
        from snep.utils import make_tables_from_path
        if statepath:
            self.prev_tables = make_tables_from_path(statepath)
            self.prev_tables.open_file(readonly=True)
            self.tables.copy_parameters(self.prev_tables, reset_results=True, new_seeds=False)

    @staticmethod
    def extract_state(job_name, resultdir):
        from snep.utils import user_select_experiment_dir
        # table = make_tables_from_path(path)
        # table.open_file(readonly=True)
        # tmpdir = mkdtemp(prefix=job_name)
        # task_ids = table.get_task_ids(onlyunfinished=False)
        # for task_id in task_ids:
        #     task_name = table.task_name(task_id)
        #     state = table.get_sim_state(task_id)
        #     task_dir = os.path.join(tmpdir, task_name)
        #     os.mkdir(task_dir)
        #     for k,v in state.iteritems():
        #         with open(os.path.join(task_dir, k), 'wb') as fh:
        #             np.savez_compressed(fh, **v)
        #             #cPickle.dump(state, fh, protocol=cPickle.HIGHEST_PROTOCOL)
        # return tmpdir
        path = os.path.expanduser('~/{resultdir}/{job_name}/'.format(resultdir=resultdir, job_name=job_name))
        print("Please choose a previous experiment to resume from: ")
        return user_select_experiment_dir(path)[0]

# def make_logger(self):
#        consoleformatter = logging.Formatter('%(levelname)s %(message)s')
#        console = logging.StreamHandler(sys.stderr)
#        console.setFormatter(consoleformatter)
# #       if subproc:
# #           console.setLevel(logging.WARN)
# #       else:
#        console.setLevel(logging.INFO)
#        
#        sg_name = 'my_experiment'
#        self.logger = 'snep.{0}'.format(sg_name)
#        
#        logger = logging.getLogger(self.logger)
#        logger.setLevel(logging.INFO)
#
#        datetime = time.strftime('%Y-%m-%d-%Hh%Mm%Ss')
#        logfile = logging.FileHandler('{0}/{1}.log'.format(logpath,datetime))
#        fileformatter = logging.Formatter('%(asctime)-6s: %(name)s: %(levelname)s %(message)s')
#        logfile.setFormatter(fileformatter)
#        logfile.setLevel(logging.INFO)
#
#        # Direct all loggers to the console and file
#        rl = logging.getLogger('')
#        rl.addHandler(logfile)
#        rl.addHandler(console)
