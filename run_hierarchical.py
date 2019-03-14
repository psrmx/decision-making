# imports and no pop-up figure from plots
import os
import numpy as np
import matplotlib as mlp
mlp.use('agg')

from snep.configuration import config
from snep.experiment import Experiment

# cluster configuration
config['cluster'] = config.run_on_cluster()
username = 'paola'
max_tasks = 50          # 260 cores in the server
mem_per_task = 64.      # in GB, do a test with 32 GB then find optimal value
max_task_time = None    # In HH:MM:SS, important if you want to jump ahead queue. For local run: None
poll_interval = 2.      # in minutes


def run_hierarchical(task_info, taskdir, tempdir):
    # dir to save results and figures
    os.mkdir(taskdir)
    print(taskdir)

    # specific imports
    import circuits as cir
    from burst_analysis import spks2neurometric
    from helper_funcs import plot_fig1, plot_fig2, plot_fig3, plot_plastic_rasters, plot_isis
    from brian2 import set_device, defaultclock, seed, profiling_summary
    from brian2.core.magic import start_scope

    # simulation parameters
    sim_dt = task_info['sim']['sim_dt']
    runtime = task_info['sim']['runtime']
    smooth_win = task_info['sim']['smooth_win']

    # parallel code and flag to start
    set_device('cpp_standalone', directory=tempdir)
    defaultclock.dt = sim_dt

    # seed to get same network with same synapses
    start_scope()
    seed(task_info['sim']['seed_con'])
    print('Creating network...')

    if task_info['sim']['plasticity']:
        net, monitors = cir.get_plasticity_net(task_info)
    else:
        net, monitors = cir.get_hierarchical_net(task_info)

    # generate stimulus
    Irec, stim1, stim2, stim_time = cir.mk_sen_stimulus(task_info, arrays=True)

    print('Running simulation...')
    net.run(runtime, report='stdout', profile=True)
    print(profiling_summary(net=net, show=10))

    # fig1 plot on cluster
    if task_info['sim']['plt_fig1']:
        mon2plt = monitors.copy() + [stim1, stim2, stim_time]
        plot_fig1(mon2plt, smooth_win, taskdir)

    # burst analysis
    if task_info['sim']['burst_analysis']:
        spksSE = monitors[0]
        events, bursts, singles, spikes, isis = spks2neurometric(task_info, spksSE, raster=True)
        plot_fig2(task_info, events, bursts, spikes, stim1, stim2, stim_time, taskdir)
        plot_isis(task_info, isis, bursts, events, taskdir)
        computed = {'events': events, 'bursts': bursts, 'singles': singles, 'spikes': spikes, 'isis': isis}
    else:
        computed = np.zeros(1)
        isis = np.zeros(1)

    # inhibitory plasticity results
    if task_info['sim']['plasticity']:
        dend_mon = monitors[-1]
        last_muOUd = np.array(dend_mon.muOUd[:, -int(5e3):].mean(axis=1))

        # plot weights
        events, bursts, singles, spikes, isis = spks2neurometric(task_info, spksSE, raster=False)
        plot_fig3(task_info, dend_mon, events, bursts, spikes, taskdir)
        plot_plastic_rasters(task_info, spksSE, bursts, isis, taskdir)
    else:
        isis = np.zeros(1)
        last_muOUd = np.zeros(1)

    # Choice selection
    # population rates and downsample

    results = {
        'raw_data': {'last_muOUd': last_muOUd},
        #              'poprates_sen': poprates_sen[:, settle_timeidx:],
        #              'pref_msk': np.array([pref_msk])
        'sim_state': np.zeros(1),
        'computed': computed
    }

    return results


class JobInfoExperiment(Experiment):
    run_task = staticmethod(run_hierarchical)

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
        from snep.utils import Parameter, ParameterArray

        param_fixed = {
            'dec': {
                'N_E': 1600,
                'N_I': 400,
                'sub': 0.15},

            'sen': {
                'N_E': 1600,
                'N_I': 400,
                'N_X': 1000,
                'sub': 0.5},

            'sim': {
                'sim_dt': Parameter(0.1, 'ms'),
                'stim_dt': Parameter(1, 'ms'),
                'runtime': Parameter(10, 'second'),
                'settle_time': Parameter(0, 'second'),
                'stim_on': Parameter(0, 'second'),
                'stim_off': Parameter(10, 'second'),
                'replicate_stim': False,
                'num_method': 'euler',
                'seed_con': Parameter(1284),
                'smooth_win': Parameter(100, 'ms'),
                'valid_burst': Parameter(16e-3),
                '2c_model': True,
                'plt_fig1': False,
                'burst_analysis': True,
                'plasticity': True},

            'plastic': {
                # 'targetB': Parameter(2, 'Hz'),
                'tauB': Parameter(50000, 'ms'),
                'tau_update': Parameter(10, 'ms'),
                'eta0': Parameter(5, 'pA'),
                'min_burst_stop': Parameter(0.1),
                'dec_winner_rate': Parameter(35, 'Hz')}}

        param_ranges = {
            'c': ParameterArray(np.array([0])),
            'bfb': ParameterArray(np.array([0])),
            'targetB': ParameterArray(np.array([2]), 'Hz'),  # np.arange(1.5, 4.5, 0.5)
            # 'iter': ParameterArray(np.arange(0, 4))
            }

        # add params to tables
        self.tables.add_parameters(param_fixed)
        self.tables.add_parameter_ranges(param_ranges)

        # link between parameters, avoids unnecessary combinations
        # self.tables.link_parameter_ranges([('tau_update',), ('B0',)])


if __name__ == '__main__':
    from snep.parallel2 import run
    """
        IMPORTANT: Only include code here that can be run repeatedly,
        because this will be run once in the parent process, and then
        once for every worker process.
    """
    # path.expanduser() may differ from result_dir
    ji_kwargs = dict(root_dir=os.path.expanduser('~/Documents/WS19/MasterThesis/Experiments'))
    job_info = run(JobInfoExperiment, ji_kwargs, username=username, max_tasks=max_tasks, mem_per_task=mem_per_task,
                   max_task_time=max_task_time, poll_interval=poll_interval,
                   result_dir='Documents/WS19/MasterThesis/Experiments',
                   additional_files=['circuits.py', 'helper_funcs.py', 'burst_analysis.py', 'get_params.py', 'neuron_models.py'])
