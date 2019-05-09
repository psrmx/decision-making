# imports and no pop-up figure from plots
import os
import numpy as np
import matplotlib as mlp
mlp.use('agg')

from snep.configuration import config
from snep.experiment import Experiment
from helper_funcs import np_array

# cluster configuration
config['cluster'] = config.run_on_cluster()
username = 'paola'
max_tasks = 50          # 260 cores in the server
mem_per_task = 20.      # in GB, do a test with 32 GB then find optimal value
max_task_time = None    # In HH:MM:SS, important if you want to jump ahead queue. For local run: None
poll_interval = 2.      # in minutes


def run_hierarchical(task_info, taskdir, tempdir):
    # dir to save results and figures
    os.mkdir(taskdir)
    print(taskdir)

    # specific imports
    import circuits as cir
    from burst_analysis import spk_mon2spk_times, spk_times2raster
    from helper_funcs import plot_fig1, plot_fig2, plot_fig3, plot_plastic_rasters, plot_isis, choice_selection, plot_plastic_check
    from brian2 import set_device, defaultclock, seed, profiling_summary, prefs
    from brian2.core.magic import start_scope

    # setup simulation
    set_device('cpp_standalone', directory=tempdir)
    prefs.core.default_float_dtype = np.float32
    sim_dt = task_info['sim']['sim_dt']
    runtime = task_info['sim']['runtime']
    defaultclock.dt = sim_dt

    # seed to get same network with same synapses
    start_scope()
    seed(task_info['sim']['seed_con'])
    print('Creating network...')

    if task_info['sim']['plasticity']:
        # task_info['sim']['smooth_win'] *= 10
        net, monitors = cir.get_plasticity_net(task_info)
    else:
        net, monitors = cir.get_hierarchical_net(task_info)

    if not task_info['sim']['online_stim']:
        Irec, stim1, stim2, stim_time = cir.mk_sen_stimulus(task_info, arrays=True)

    print('Running simulation...')
    net.run(runtime, report='stdout', profile=True)
    print(profiling_summary(net=net, show=10))

    if task_info['sim']['online_stim']:
        # retrieve stim monitor info
        stim_mon = monitors[2]
        sub = int(stim_mon.source.__len__() / 2)
        stim_time = stim_mon.t_
        stim1 = stim_mon.I[:sub]
        stim2 = stim_mon.I[sub:]

    # results
    raw_data = np.zeros(1, dtype=np.float32)
    computed = np.zeros(1, dtype=np.float32)

    if task_info['sim']['plasticity']:
        spksSE = monitors[0]
        dend_mon = monitors[1]
        spks_dend = monitors[-2]
        pop_dend = monitors[-1]
        last_muOUd = np_array(dend_mon.muOUd[:, -int(10e3):-int(5e3)].mean(axis=1))  # last 10:5 sec
        all_spk_times, _ = spk_mon2spk_times(task_info, spksSE)
        events, bursts, singles, spikes = spk_times2raster(task_info, all_spk_times, broad_step=True, rate=True)
        plot_fig3(task_info, dend_mon, events, bursts, spikes, pop_dend, taskdir)
        plot_plastic_rasters(task_info, all_spk_times[3], all_spk_times[1], bursts, taskdir)
        plot_plastic_check(task_info, pop_dend, spks_dend, bursts, all_spk_times[1], taskdir)

        raw_data = {'last_muOUd': last_muOUd}
        computed = {'events': all_spk_times[0], 'bursts': all_spk_times[1], 'singles': all_spk_times[2],
                    'spikes': all_spk_times[3]}

    else:
        # choice selection
        choice_monitors = monitors[1:5]
        raw_data = choice_selection(task_info, choice_monitors)

        if task_info['sim']['plt_fig1']:
            mon2plt = monitors.copy() + [stim1, stim2, stim_time]
            plot_fig1(task_info, mon2plt, taskdir)

        if task_info['sim']['burst_analysis']:
            spksSE = monitors[0]
            all_spk_times, all_isis = spk_mon2spk_times(task_info, spksSE)
            events, bursts, singles, spikes, downsample = spk_times2raster(task_info, all_spk_times,
                                                                           broad_step=True, downsample=True)
            plot_fig2(task_info, events, bursts, spikes, stim1, stim2, stim_time, taskdir)
            plot_isis(task_info, *all_isis, task_dir=taskdir)

            computed = {'events': events, 'bursts': bursts, 'singles': singles, 'spikes': spikes,
                        'events_low_def': downsample[0], 'bursts_low_def': downsample[1],
                        'singles_low_def': downsample[2], 'spikes_low_def': downsample[3],
                        'isis': all_isis[0], 'ieis': all_isis[1], 'ibis': all_isis[2],
                        'cvs': all_isis[3], 'spks_per_burst': all_isis[4]}

    results = {
        'raw_data': raw_data,
        'sim_state': np.zeros(1, dtype=np.float32),
        'computed': computed}

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
                'runtime': Parameter(25, 'second'),
                'settle_time': Parameter(0, 'second'),
                'stim_on': Parameter(1, 'second'),
                'stim_off': Parameter(25, 'second'),
                'replicate_stim': False,
                'num_method': 'euler',
                'seed_con': Parameter(1284),
                'smooth_win': Parameter(1000, 'ms'),
                'valid_burst': Parameter(16e-3),
                'cp_step': 10,
                '2c_model': False,
                'plt_fig1': False,
                'burst_analysis': True,
                'plasticity': False,
                'online_stim': False},

            'plastic': {
                'tauB': Parameter(50000, 'ms'),
                'tau_update': Parameter(10, 'ms'),
                'eta0': Parameter(1, 'pA'),
                'min_burst_stop': Parameter(0.1),
                'dec_winner_rate': Parameter(50, 'Hz')}}

        param_ranges = {
            'c': ParameterArray(np_array([0])),     # np.linspace(-1, 1, 11)
            'bfb': ParameterArray(np_array([1])),
            'targetB': ParameterArray(np_array([2]), 'Hz'),  # np.arange(1.5, 4.5, 0.5)
            'iter': ParameterArray(np.arange(0, 1))
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
                   additional_files=['circuits.py', 'neuron_models.py', 'get_params.py', 'helper_funcs.py',
                                     'burst_analysis.py', 'choice_analysis.py'])
