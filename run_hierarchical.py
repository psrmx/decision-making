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


def plot_fig1b(monitors, win, taskdir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from brian2.units import second
    from brian2tools import plot_raster, plot_rate
    sns.set(context='paper', style='darkgrid')

    spksSE, spksDE, rateDE1, rateDE2, rateDI, rateSE1, rateSE2, rateSI, stim1, stim2, stimtime = monitors
    subDE = int(spksDE.source.__len__() / 2)
    subSE = int(spksSE.source.__len__() / 2)
    nticks = 4

    fig1, axs = plt.subplots(7, 1, figsize=(8, 12), sharex=True)
    # decision circuit
    fig1.add_axes(axs[0])
    plt.title('Decision circuit')
    spksDE2 = spksDE.i >= subDE
    plot_raster(spksDE.i[spksDE2], spksDE.t[spksDE2], color='C0', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(subDE, 2*subDE+1, subDE/nticks))
    plt.ylabel('neuron index')

    fig1.add_axes(axs[1])
    pos = axs[1].get_position()
    axs[1].set_position([pos.x0, pos.y0 + .01, pos.width, pos.height])
    spksDE1 = spksDE.i < subDE
    plot_raster(spksDE.i[spksDE1], spksDE.t[spksDE1], color='C3', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(0, subDE+1, subDE/nticks))
    plt.ylabel('neuron index')  # , {'horizontalalignment':'right'})

    fig1.add_axes(axs[2])
    pos = axs[2].get_position()
    axs[2].set_position([pos.x0, pos.y0 + .02, pos.width, pos.height])
    plot_rate(rateDI.t, rateDI.smooth_rate(window='flat', width=win), color='C4', time_unit=second, lw=1, label='inh')
    plot_rate(rateDE1.t, rateDE1.smooth_rate(window='flat', width=win), color='C3', time_unit=second, lw=1.5,
              label='exc1')
    plot_rate(rateDE2.t, rateDE2.smooth_rate(window='flat', width=win), color='C0', time_unit=second, lw=1.5,
              label='exc2')
    plt.ylabel("rate (sp/s)")
    plt.ylim(0, 45)
    plt.yticks(np.arange(0, 55, 20))
    plt.legend(loc='upper left', fontsize='x-small')

    # sensory circuit
    fig1.add_axes(axs[3])
    plt.title('Sensory circuit')
    spksSE2 = spksSE.i >= subSE
    plot_raster(spksSE.i[spksSE2], spksSE.t[spksSE2], color='C0', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(subSE, 2*subSE+1, subSE/nticks))
    plt.ylabel('neuron index')

    fig1.add_axes(axs[4])
    pos = axs[4].get_position()
    axs[4].set_position([pos.x0, pos.y0 + .01, pos.width, pos.height])
    spksSE1 = spksSE.i < subSE
    plot_raster(spksSE.i[spksSE1], spksSE.t[spksSE1], color='C3', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(0, subSE+1, subSE/nticks))
    plt.ylabel('neuron index')  # , {'horizontalalignment':'right'})

    fig1.add_axes(axs[5])
    pos = axs[5].get_position()
    axs[5].set_position([pos.x0, pos.y0 + .02, pos.width, pos.height])
    plot_rate(rateSI.t, rateSI.smooth_rate(window='flat', width=win), color='C4', time_unit=second, lw=1, label='inh')
    plot_rate(rateSE1.t, rateSE1.smooth_rate(window='flat', width=win), color='C3', time_unit=second, lw=1.5,
              label='exc1')
    plot_rate(rateSE2.t, rateSE2.smooth_rate(window='flat', width=win), color='C0', time_unit=second, lw=1.5,
              label='exc2')
    plt.ylabel("rate (sp/s)")
    plt.ylim(0, 25)
    plt.yticks(np.arange(0, 30, 10))
    plt.legend(loc='upper left', fontsize='x-small')

    # Stimulus
    fig1.add_axes(axs[6])
    plt.title('Stimulus')
    plt.plot(stimtime, stim1.mean(axis=0)*1e12, color='C3', lw=1.5)   # stim1.t, axis=0
    plt.plot(stimtime, stim2.mean(axis=0)*1e12, color='C0', lw=1.5)   # np.arange(0, 3.5, 1e-3), axis=1
    plt.xlabel("time (s)")
    plt.ylabel("current (pA)")
    #plt.xlim(0, 5)

    # xlabels
    for i in range(6):
        axs[i].set_xlabel('')

    fig1.savefig(taskdir + '/figure1.png')
    plt.close(fig1)


def run_hierarchical(task_info, taskdir, tempdir):
    # dir to save results and figures
    os.mkdir(taskdir)
    print(taskdir)

    # specific imports
    import circuits as cir
    from brian2 import set_device, defaultclock, seed, Network, profiling_summary
    from brian2.core.magic import start_scope
    from helper_funcs import unitless

    # simulation parameters
    sim_dt = task_info['sim']['sim_dt']
    runtime = task_info['sim']['runtime']
    settle_time = unitless(task_info['sim']['settle_time'], sim_dt)
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

    # nice plots on cluster
    if task_info['sim']['plt_fig1']:
        mon2plt = monitors.copy() + [stim1, stim2, stim_time]
        plot_fig1b(mon2plt, smooth_win, taskdir)

    # Burst analysis
    events = np.zeros(1)
    bursts = np.zeros(1)
    singles = np.zeros(1)
    spikes = np.zeros(1)
    last_muOUd = np.zeros(1)

    if task_info['sim']['burst_analysis']:
        pass

    # neurometric params

    # -------------------------------------
    # Choice selection
    # -------------------------------------
    # population rates and downsample

    results = {
        'raw_data': np.zeros(1),
        #              'poprates_sen': poprates_sen[:, settle_timeidx:],
        #              'pref_msk': np.array([pref_msk]),
        #              'last_muOUd': last_muOUd},
        'sim_state': np.zeros(1),
        'computed': {'events': events, 'bursts': bursts, 'singles': singles, 'spikes': spikes}}
        #, 'isis': np.array(isis)}}

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
                'runtime': Parameter(5, 'second'),
                'settle_time': Parameter(0, 'second'),
                'stim_on': Parameter(0, 'second'),
                'stim_off': Parameter(5, 'second'),
                'replicate_stim': False,
                'num_method': 'euler',
                'seed_con': Parameter(1284),
                'smooth_win': Parameter(100, 'ms'),
                '2c_model': True,
                'plt_fig1': False,
                'burst_analysis': False,
                'plasticity': True},

            'plastic': {
                # 'targetB': Parameter(2, 'Hz'),
                'tauB': Parameter(50000, 'ms'),
                'tau_update': Parameter(10, 'ms'),
                'eta0': Parameter(5, 'pA'),
                'valid_burst': Parameter(16e-3),  # in seconds
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
