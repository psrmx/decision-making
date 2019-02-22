import matplotlib as mlp
mlp.use('agg')

from snep.configuration import config
from snep.experiment import Experiment
import os
import numpy as np

config['cluster'] = config.run_on_cluster()

username = 'paola'
max_tasks = 50  # 260 cores in the server
mem_per_task = 128.  # in GB, do a test with 32 GB then find paramspace/results_map/maxvmem
# this is the max memory of your run. ask for sth slightly higher, as a safe cushion.
max_task_time = None  # In HH:MM:SS, generally not required, but important if you want to have a chance of
# cutting the line in the server queue. For running locally specify None!
poll_interval = 2.  # in minutes


# Helper functions
def get_OUstim(n, tau):
    from scipy.signal import lfilter
    # Ornstein-Uhlenbeck process in discrete time
    a = np.exp(-(1 / tau))
    i = lfilter(np.ones(1), [1, -a], np.sqrt(1 - a * a) * np.random.randn(n))
    return i


def generate_stim(nn, stimdur, taustim):
    """
    Generate common and private part of the stimuli from OU process
    :param nn: number of neurons per subpop
    :param stimdur: stimulus duration (i.e. timepoints)
    :param taustim: tau constant of stim
    :return: array of shape nn * tp
    """
    from brian2.units import ms

    tp = int(stimdur / ms)
    tau = int(taustim / ms)

    # common
    z1 = get_OUstim(tp, tau)
    z1 = np.tile(z1, (nn, 1))
    z2 = get_OUstim(tp, tau)
    z2 = np.tile(z2, (nn, 1))

    # private
    zk1 = get_OUstim(tp * nn, tau)
    zk1 = np.asarray(zk1).reshape(nn, tp)
    zk2 = get_OUstim(tp * nn, tau)
    zk2 = np.asarray(zk2).reshape(nn, tp)

    return z1, z2, zk1, zk2


def smoothpoprate(raster, sub, win=50e-3, dt=1e-3):
    """

    :param raster: a 2D array containig spiketimes of spikes, events or bursts
    :param sub: index that represents end of pop1 and beginnig of pop2
    :param dt:  integration time step
    :param win: width of kernel window
    :return: the smoothed rate of pop1 and pop2
    """
    # population rates by averaging and binning
    popraster1 = np.squeeze(raster[:sub].mean(axis=0)) / win
    popraster2 = np.squeeze(raster[sub:].mean(axis=0)) / win

    # smooth rate
    kernel = np.ones((int(win / dt)))
    rate1 = np.convolve(popraster1, kernel, mode='same')
    rate2 = np.convolve(popraster2, kernel, mode='same')

    # variables2return
    rate = np.vstack((rate1, rate2))
    sub = int(rate.shape[0] /2)

    return rate, sub


def plot_fig1b(monitors, win, taskdir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from brian2.units import second
    from brian2tools import plot_raster, plot_rate
    sns.set(context='paper', style='darkgrid')

    rateDE1, rateDE2, rateDI, spksDE, rateSE1, rateSE2, rateSI, spksSE, stim1, stim2, stimtime = monitors
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
    #plt.plot(statSI.I.mean(axis=0), color='C4', lw=1)
    plt.xlabel("time (s)")
    plt.ylabel("current (pA)")
    #plt.xlim(0, 5)

    # xlabels
    for i in range(6):
        axs[i].set_xlabel('')

    fig1.savefig(taskdir + '/figure1.png')
    plt.close(fig1)


def plot_neurometric(events, bursts, spikes, stim1, stim2, stimtime, time01, taskdir, smoothwin=50e-3):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # params from dimensions
    subSE = int(events.shape[0] / 2)
    time = np.linspace(time01[0], time01[1], events.shape[1])

    # obtain rates if events, bursts are rasters
    if subSE > 1:
        spikes, _ = smoothpoprate(spikes, subSE, win=smoothwin)
        events, _ = smoothpoprate(events, subSE, win=smoothwin)
        bursts, subSE = smoothpoprate(bursts, subSE, win=smoothwin)

    # start figure
    fig2, axs = plt.subplots(5, 2, figsize=(16, 10), sharex='row', sharey='row')

    # stimulus
    axs[0, 0].plot(stimtime, stim1.mean(axis=0)*1e12, color='C3', lw=1.5)
    axs[0, 0].set_title('sensory population 1')
    axs[0, 0].set_ylabel(r'$som_{input}$ (pA)')
    axs[0, 0].set_xlim(time[0]-0.2, time[-1]+0.2)
    axs[0, 1].plot(stimtime, stim2.mean(axis=0)*1e12, color='C0', lw=1.5)
    axs[0, 1].set_title('sensory population 2')

    # firing rate
    axs[1, 0].plot(time, spikes[:subSE].mean(axis=0), lw=1.5, color='C5')
    axs[1, 0].set_ylabel(r'$A$ (Hz)')
    axs[1, 1].plot(time, spikes[subSE:].mean(axis=0), lw=1.5, color='C5')
    # plt.ylim(0, 25)

    # burst rate
    axs[2, 0].plot(time, bursts[:subSE].mean(axis=0), lw=1.5, color='C1')
    axs[2, 0].set_ylabel(r'$B$ (Hz)')
    axs[2, 1].plot(time, bursts[subSE:].mean(axis=0), lw=1.5, color='C1')

    # burst fraction
    bfracc1 = bursts[:subSE].mean(axis=0) / events[:subSE].mean(axis=0)
    bfracc1[np.isnan(bfracc1)] = 0  # handle division by zero
    bfracc2 = bursts[subSE:].mean(axis=0) / events[subSE:].mean(axis=0)
    bfracc2[np.isnan(bfracc2)] = 0  # handle division by zero
    axs[3, 0].plot(time, bfracc1*100, lw=1.5, color='C6')
    axs[3, 0].set_ylabel(r'$F$ (%)')
    axs[3, 1].plot(time, bfracc2*100, lw=1.5, color='C6')

    # event rate
    axs[4, 0].plot(time, events[:subSE].mean(axis=0), lw=1.5, color='C2')
    axs[4, 0].set_ylabel(r'$E$ (Hz)')
    axs[4, 0].set_xlabel(r'$time$ (ms)')
    axs[4, 1].plot(time, events[subSE:].mean(axis=0), lw=1.5, color='C2')
    axs[4, 1].set_xlabel(r'$time$ (ms)')

    fig2.savefig(taskdir + '/figure2.png')
    plt.close(fig2)


def plot_isis(isis, bursts, events, time01, taskdir, maxisi=500):
    """
    bursts and events are rasters (0 or 1)
    :param isis:
    :param bursts:
    :param events:
    :param time01:
    :param taskdir:
    :param maxisi:
    :return:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='paper', style='darkgrid')

    # params from dimensions
    n, timepts = events.shape
    time = np.linspace(time01[0], time01[1], timepts)
    time = np.repeat(time[np.newaxis, :], n, axis=0)

    # get different versions of isis
    isis_b = np.diff(time[bursts.astype(bool)].reshape(n, -1), axis=1).flatten()
    isis_e = np.diff(time[events.astype(bool)].reshape(n, -1), axis=1).flatten()
    cv = isis.std() / isis.mean()
    cv_b = isis_b.std() / isis_b.mean()
    cv_e = isis_e.std() / isis_e.mean()

    # start figure
    fig3, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=False)

    # all isis
    sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C5', ax=axs[0])
    axs[0].plot(0, c='white', label=r'CV = %.3f' % cv)
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel(r'$Proportion$ $of$ $isi$')

    # burst isis
    sns.distplot(isis_b, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C1', ax=axs[1])
    axs[1].plot(0, c='white', label=r'CV_b = %.3f' % cv_b)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Interspike interval distribution')
    axs[1].set_xlabel(r'$Interspike$ $interval$ (ms)')
    axs[1].set_xlim(0, maxisi)

    # event isis
    sns.distplot(isis_e, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2', ax=axs[2])
    axs[2].plot(0, c='white', label=r'CV_e = %.3f' % cv_e)
    axs[2].legend(loc='upper right')

    # save figure
    plt.tight_layout()
    fig3.savefig(taskdir + '/figure3.png')
    plt.close(fig3)


def create_inset(axes, data2plt, c, xlim, w=1, h=0.7, nyticks=4):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(axes, w, h, loc=1)
    axins.plot(data2plt[0], data2plt[1], color=c, lw=1.5)
    axins.set_xlim(xlim)
    axins.yaxis.get_major_locator().set_params(nbins=nyticks)
    plt.xticks(visible=False)

    return axins


def plot_weights(mon, events, bursts, spikes, infosimul, taskdir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from brian2.units import Hz, pA, ms
    sns.set(context='paper', style='ticks')

    # params
    target, B0, eta, tauB, tau_update, smoothwin = infosimul
    subSE = int(events.shape[0] / 2)
    time = np.linspace(0, mon.t_[-1], bursts.shape[1])
    lasttime = time[-1]
    zoomt = (lasttime-1, lasttime)
    dt = lasttime / bursts.shape[1]
    target /= Hz
    eta /= pA
    tauB /= ms
    tau_update /= ms
    nn2plt = 10

    fig2, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig2.add_axes(axs[0, 0])
    plt.title(r'Drift of dendritic noise ($\mu_{OU_{d}}$)')
    plt.plot(mon.t_, mon.muOUd[:nn2plt].T*1e12, color='gray', lw=0.5)
    plt.plot(mon.t_, mon.muOUd.mean(axis=0)*1e12, color='C6', lw=1.5)
    plt.ylabel(r'$weights$ $(pA)$')
    create_inset(axs[0, 0], (mon.t_, mon.muOUd.mean(axis=0)*1e12), 'C6', zoomt)

    fig2.add_axes(axs[1, 0])
    plt.title(r'Noise background current to dendrites ($I_{OU_{d}}$)')
    plt.plot(mon.t_, mon.Ibg[:nn2plt].T*1e9, color='gray', lw=0.5)
    plt.plot(mon.t_, mon.Ibg.mean(axis=0)*1e9, color='C0', lw=1.5)
    plt.ylabel('$I_{OU_{d}}$ $(nA)$')
    create_inset(axs[1, 0], (mon.t_, mon.Ibg.mean(axis=0)*1e9), 'C0', zoomt)

    fig2.add_axes(axs[2, 0])
    plt.title(r'Excitation current from integration circuit')
    plt.plot(mon.t_, mon.g_ea[:nn2plt].T, color='gray', lw=0.5)
    plt.plot(mon.t_, mon.g_ea.mean(axis=0), color='C3', lw=1.5)
    plt.xlabel(r'$Time$ (s)')
    plt.ylabel(r'$g_{AMPA}$ $(~nS)$')
    create_inset(axs[2, 0], (mon.t_, mon.g_ea.mean(axis=0)), 'C3', zoomt)

    fig2.add_axes(axs[0, 1])
    plt.title('PSTH')
    plt.plot(time, spikes[:subSE].mean(axis=0), lw=1.5, color='C5')
    plt.ylabel(r'$Firing$ $rate$ $(Hz)$')
    create_inset(axs[0, 1], (time, spikes[:subSE].mean(axis=0)), 'C5', zoomt)

    fig2.add_axes(axs[1, 1])
    #plt.title('Burst rate')
    plt.plot(time, bursts[:subSE].mean(axis=0), lw=1.5, color='C1')
    plt.axhline(target, color='gray', lw=2, ls='dashed')
    plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
    #plt.ylim(-0.5, 5.5)
    create_inset(axs[1, 1], (time, bursts[:subSE].mean(axis=0)), 'C1', zoomt)

    fig2.add_axes(axs[2, 1])
    bfracc = bursts[:subSE].mean(axis=0) / events[:subSE].mean(axis=0)
    bfracc[np.isnan(bfracc)] = 0    # handle division by 0 - because no event also means no burst!
    # epsilon = 1e-5    # +epsilon avoids division by 0, but it's not needed here.
    #plt.title('Burst fraction')
    plt.plot(time, bfracc*100, lw=1.5, color='C2')
    plt.ylabel(r'$Burst$ $fraction$ $(\%)$')
    plt.xlabel(r'$Time$ $(s)$')
    plt.ylim(0, 100)
    create_inset(axs[2, 1], (time, bfracc*100), 'C2', zoomt)

    fig2.add_axes(axs[0, 2])
    B = mon.B.mean(axis=0)
    plt.plot(mon.t_, mon.B[:nn2plt].T - B0, color='gray', lw=0.5)
    plt.plot(mon.t_, B - B0, color='C4', lw=1.5)
    plt.ylabel(r'$B - B0$')
    create_inset(axs[0, 2], (mon.t_, B-B0), 'C4', zoomt)

    fig2.add_axes(axs[1, 2])
    plt.title('Integral of B in a 10 sec span')
    if lasttime > 50:
        a1_start = int(10/dt)               # from 10:20 sec
        a2_start = int((time[-1]/2)/dt)       # middle 10 sec
        a3_start = int((time[-1]-10)/dt)    # last 10 sec
        step = a1_start
        area1 = B[a1_start:a1_start + step].sum() * dt
        area2 = B[a2_start:a2_start + step].sum() * dt
        area3 = B[a3_start:a3_start + step].sum() * dt

        plt.plot(time[a1_start], np.abs(area1), 'o', label='initial')
        plt.plot(time[a2_start], np.abs(area2), 'o', label='middle')
        plt.plot(time[a3_start], np.abs(area3), 'o', label='late')
        plt.legend(loc='best')
    plt.ylabel(r'$AUC$ $(au)$')

    fig2.add_axes(axs[2, 2])
    plt.title('Info of simulation')
    plt.plot(0, c='white', label=r'target = %.1f Hz' % target)
    plt.plot(0, c='white', label=r'eta = %.3f pA' % eta)
    plt.plot(0, c='white', label=r'tauB = %i ms' % tauB)
    plt.plot(0, c='white', label=r'tau_update = %i ms' % tau_update)
    plt.legend(loc='best', fontsize='large', frameon=False)
    plt.xlabel(r'$Time$ $(s)$')

    #plt.tight_layout()
    fig2.savefig(taskdir + '/figure2.png')
    plt.close(fig2)


def plot_rasters(spkmon, bursts, target, isis, lasttime, taskdir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from brian2.units import second
    from brian2tools import plot_raster
    sns.set(context='paper', style='darkgrid')

    # params
    subSE = min(int(spkmon.source.__len__() / 2), 20)
    nticks = 4

    fig3, axs = plt.subplots(4, 1, figsize=(4, 12), sharex=False)

    # sensory circuit
    fig3.add_axes(axs[0])
    plt.title('Before plasticity')
    spksSE1 = spkmon.i < subSE
    plot_raster(spkmon.i[spksSE1], spkmon.t[spksSE1], color='gray', marker='.', markersize=0.7, time_unit=second)
    plt.yticks(np.arange(0, subSE+1, subSE/nticks))
    plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
    plt.xlim(5, 10)
    plt.xlabel('')

    fig3.add_axes(axs[1])
    plt.title('After plasticity')
    plot_raster(spkmon.i[spksSE1], spkmon.t[spksSE1], color='C5', marker='.', markersize=0.7, time_unit=second)
    plt.yticks(np.arange(0, subSE+1, subSE/nticks))
    plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
    plt.xlim(lasttime-5, lasttime)
    plt.xlabel(r'$Times$ $(s)$')

    fig3.add_axes(axs[2])
    maxbrate = 5.5
    plt.plot(target, bursts[:subSE].mean(axis=0)[int(lasttime-10):].mean(), c='C1', marker='o')
    plt.plot(np.linspace(0, maxbrate, 100), np.linspace(0, maxbrate, 100), c='gray', lw=1.5, ls='dashed')
    plt.xlabel(r'$Target$ $rate$ $(Hz)$')
    plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
    plt.xlim(0, 5.5)
    plt.ylim(0, 5.5)

    fig3.add_axes(axs[3])
    maxisi = 500
    cv = isis.std() / isis.mean()
    sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2')
    plt.plot(0, c='white', label=r'CV = %.3f' % cv)
    plt.title('Interspike interval distribution')
    plt.xlabel(r'$Interspike$ $interval$ $(ms)$')
    plt.ylabel(r'$Percent$ $of$ $isi$')
    plt.xlim(0, maxisi)
    plt.legend(loc='best')

    plt.tight_layout()
    fig3.savefig(taskdir + '/figure3.png')
    plt.close(fig3)


def run_task_hierarchical(task_info, taskdir, tempdir):
    # imports
    from brian2 import defaultclock, set_device, seed, TimedArray, Network, profiling_summary
    from brian2.monitors import SpikeMonitor, PopulationRateMonitor, StateMonitor
    from brian2.synapses import Synapses
    from brian2.core.magic import start_scope
    from brian2.units import second, ms, amp
    from integration_circuit import mk_intcircuit
    from sensory_circuit import mk_sencircuit, mk_sencircuit_2c, mk_sencircuit_2cplastic
    from burstQuant import spks2neurometric
    from scipy import interpolate

    # if you want to put something in the taskdir, you must create it first
    os.mkdir(taskdir)
    print(taskdir)

    # parallel code and flag to start
    set_device('cpp_standalone', directory=tempdir)
    #prefs.devices.cpp_standalone.openmp_threads = max_tasks
    start_scope()

    # simulation parameters
    seedcon = task_info['simulation']['seedcon']
    runtime = task_info['simulation']['runtime']
    runtime_ = runtime / second
    settletime = task_info['simulation']['settletime']
    settletime_ = settletime / second
    stimon = task_info['simulation']['stimon']
    stimoff = task_info['simulation']['stimoff']
    stimoff_ = stimoff / second
    stimdur = stimoff - stimon
    smoothwin = task_info['simulation']['smoothwin']
    nummethod = task_info['simulation']['nummethod']

    # -------------------------------------
    # Construct hierarchical network
    # -------------------------------------
    # set connection seed
    seed(seedcon)  # set specific seed to test the same network, this way we also have the same synapses!

    # decision circuit
    Dgroups, Dsynapses, Dsubgroups = mk_intcircuit(task_info)
    decE = Dgroups['DE']
    decI = Dgroups['DI']
    decE1 = Dsubgroups['DE1']
    decE2 = Dsubgroups['DE2']

    # sensory circuit, ff and fb connections
    eps = 0.2           # connection probability
    d = 1 * ms          # transmission delays of E synapses
    if task_info['simulation']['2cmodel']:
        if task_info['simulation']['plasticdend']:
            # plasticity rule in dendrites --> FB synapses will be removed from the network!
            Sgroups, Ssynapses, Ssubgroups = mk_sencircuit_2cplastic(task_info)

        else:
            # 2c model (Naud)
            Sgroups, Ssynapses, Ssubgroups = mk_sencircuit_2c(task_info)

        senE = Sgroups['soma']
        dend = Sgroups['dend']
        senI = Sgroups['SI']
        senE1 = Ssubgroups['soma1']
        senE2 = Ssubgroups['soma2']
        dend1 = Ssubgroups['dend1']
        dend2 = Ssubgroups['dend2']

        # FB
        wDS = 0.003  # synaptic weight of FB synapses, 0.0668 nS when scaled by gleakE of sencircuit_2c
        synDE1SE1 = Synapses(decE1, dend1, model='w : 1', method=nummethod, on_pre='x_ea += w', delay=d)
        synDE2SE2 = Synapses(decE2, dend2, model='w : 1', method=nummethod, on_pre='x_ea += w', delay=d)

    else:
        # normal sensory circuit (Wimmer)
        Sgroups, Ssynapses, Ssubgroups = mk_sencircuit(task_info)
        senE = Sgroups['SE']
        senI = Sgroups['SI']
        senE1 = Ssubgroups['SE1']
        senE2 = Ssubgroups['SE2']

        # FB
        wDS = 0.004  # synaptic weight of FB synapses, 0.0668 nS when scaled by gleakE of sencircuit
        synDE1SE1 = Synapses(decE1, senE1, model='w : 1', method=nummethod, on_pre='x_ea += w', delay=d)
        synDE2SE2 = Synapses(decE2, senE2, model='w : 1', method=nummethod, on_pre='x_ea += w', delay=d)

    # feedforward synapses from sensory to integration
    wSD = 0.0036        # synaptic weight of FF synapses, 0.09 nS when scaled by gleakE of intcircuit
    synSE1DE1 = Synapses(senE1, decE1, model='w : 1', method=nummethod, on_pre='g_ea += w', delay=d)
    synSE1DE1.connect(p='eps')
    synSE1DE1.w = 'wSD'
    synSE2DE2 = Synapses(senE2, decE2, model='w : 1', method=nummethod, on_pre='g_ea += w', delay=d)
    synSE2DE2.connect(p='eps')
    synSE2DE2.w = 'wSD'

    # feedback synapses from integration to sensory
    b_fb = task_info['bfb']            # feedback strength, between 0 and 6
    wDS *= b_fb     # synaptic weight of FB synapses, 0.0668 nS when scaled by gleakE of sencircuit
    synDE1SE1.connect(p='eps')
    synDE1SE1.w = 'wDS'
    synDE2SE2.connect(p='eps')
    synDE2SE2.w = 'wDS'

    # -------------------------------------
    # Create stimuli
    # -------------------------------------
    if task_info['stimulus']['replicate']:
        # replicated stimuli across iters()
        np.random.seed(task_info['seed'])  # numpy seed for OU process
    else:
        # every trials has different stimuli
        np.random.seed()
    # Note that in standalone we need to specify np seed because it's not taken care with Brian's seed() function!

    if task_info['simulation']['2cmodel']:
        I0 = task_info['stimulus']['I0s']
        last_muOUd = np.loadtxt("last_muOUd.csv")  # save the mean
    else:
        I0 = task_info['stimulus']['I0']    # mean input current for zero-coherence stim
    c = task_info['c']      # stim coherence (between 0 and 1)
    mu1 = task_info['stimulus']['mu1']  # av. additional input current to senE1 at highest coherence (c=1)
    mu2 = task_info['stimulus']['mu2']  # av. additional input current to senE2 at highest coherence (c=1)
    sigma = task_info['stimulus']['sigma']  # amplitude of temporal modulations of stim
    sigmastim = 0.212 * sigma   # std of modulation of stim inputs
    sigmaind = 0.212 * sigma    # std of modulations in individual inputs
    taustim = task_info['stimulus']['taustim']  # correlation time constant of Ornstein-Uhlenbeck process

    # generate stim from OU process
    N_stim = int(senE1.__len__())
    z1, z2, zk1, zk2 = generate_stim(N_stim, stimdur, taustim)

    # stim2exc
    i1 = I0 * (1 + c * mu1 + sigmastim * z1 + sigmaind * zk1)
    i2 = I0 * (1 + c * mu2 + sigmastim * z2 + sigmaind * zk2)
    stim_dt = 1*ms
    i1t = np.concatenate((np.zeros((int(stimon / ms), N_stim)), i1.T, np.zeros((int((runtime - stimoff) / stim_dt), N_stim))),
                         axis=0)
    i2t = np.concatenate((np.zeros((int(stimon / ms), N_stim)), i2.T, np.zeros((int((runtime - stimoff) / stim_dt), N_stim))),
                         axis=0)
    Irec = TimedArray(np.concatenate((i1t, i2t), axis=1)*amp, dt=stim_dt)

    # -------------------------------------
    # Simulation
    # -------------------------------------
    # set initial conditions (different for evert trial)
    seed()
    decE.g_ea = '0.2 * rand()'
    decI.g_ea = '0.2 * rand()'
    decE.V = '-52*mV + 2*mV * rand()'
    decI.V = '-52*mV + 2*mV * rand()'   # random initialization near 0, prevent an early decision!
    senE.g_ea = '0.05 * (1 + 0.2*rand())'
    senI.g_ea = '0.05 * (1 + 0.2*rand())'
    senE.V = '-52*mV + 2*mV*rand()'     # random initialization near Vt, avoid initial bump!
    senI.V = '-52*mV + 2*mV*rand()'

    if task_info['simulation']['2cmodel']:
        dend.g_ea = '0.05 * (1 + 0.2*rand())'
        dend.V_d = '-72*mV + 2*mV*rand()'
        dend.muOUd = np.tile(last_muOUd, 2) * amp

    # create monitors
    rateDE1 = PopulationRateMonitor(decE1)
    rateDE2 = PopulationRateMonitor(decE2)

    rateSE1 = PopulationRateMonitor(senE1)
    rateSE2 = PopulationRateMonitor(senE2)
    subSE = int(senE1.__len__())
    spksSE = SpikeMonitor(senE[subSE-100:subSE+100])  # last 100 of SE1 and first 100 of SE2

    # construct network
    net = Network(Dgroups.values(), Dsynapses.values(),
                  Sgroups.values(), Ssynapses.values(),
                  synSE1DE1, synSE2DE2, synDE1SE1, synDE2SE2,
                  rateDE1, rateDE2, rateSE1, rateSE2,
                  spksSE, name='hierarchicalnet')

    # create more monitors for plot
    if task_info['simulation']['pltfig1']:
        # inh
        rateDI = PopulationRateMonitor(decI)
        rateSI = PopulationRateMonitor(senI)

        # spk monitors
        subDE = int(decE1.__len__() * 2)
        spksDE = SpikeMonitor(decE[:subDE])
        spksSE = SpikeMonitor(senE)

        # state mons no more, just the arrays
        stim1 = i1t.T
        stim2 = i2t.T
        stimtime = np.linspace(0, runtime_, stim1.shape[1])

        # construct network
        net = Network(Dgroups.values(), Dsynapses.values(),
                      Sgroups.values(), Ssynapses.values(),
                      synSE1DE1, synSE2DE2, synDE1SE1, synDE2SE2,
                      spksDE, rateDE1, rateDE2, rateDI,
                      spksSE, rateSE1, rateSE2, rateSI,
                      name='hierarchicalnet')

    if task_info['simulation']['plasticdend']:
        # create state monitor to follow muOUd and add it to the networks
        dend_mon = StateMonitor(dend1, variables=['muOUd', 'Ibg', 'g_ea', 'B'], record=True, dt=1*ms)
        net.add(dend_mon)

        # remove FB synapses!
        net.remove([synDE1SE1, synDE2SE2, Dsynapses.values()])
        print("   FB synapses and synapses of decision circuit are ignored in this simulation!")

    # run hierarchical net
    net.run(runtime, report='stdout', profile=True)
    print(profiling_summary(net=net, show=10))

    # nice plots on cluster
    if task_info['simulation']['pltfig1']:
        plot_fig1b([rateDE1, rateDE2, rateDI, spksDE, rateSE1, rateSE2, rateSI, spksSE, stim1, stim2, stimtime],
                    smoothwin, taskdir)

    # -------------------------------------
    # Burst quantification
    # -------------------------------------
    events = np.zeros(1)
    bursts = np.zeros(1)
    singles = np.zeros(1)
    spikes = np.zeros(1)
    last_muOUd = np.zeros(1)

    # neurometric params
    dt = spksSE.clock.dt
    validburst = task_info['sen']['2c']['validburst']
    smoothwin_ = smoothwin / second

    if task_info['simulation']['burstanalysis']:

        if task_info['simulation']['2cmodel']:
            last_muOUd = np.array(dend_mon.muOUd[:, -int(1e3):].mean(axis=1))

        if task_info['simulation']['plasticdend']:
            # calculate neurometric info per population
            events, bursts, singles, spikes, isis = spks2neurometric(spksSE, runtime, settletime, validburst,
                                                                     smoothwin=smoothwin_, raster=False)

            # plot & save weigths after convergence
            eta0 = task_info['sen']['2c']['eta0']
            tauB = task_info['sen']['2c']['tauB']
            targetB = task_info['targetB']
            B0 = tauB * targetB
            tau_update = task_info['sen']['2c']['tau_update']
            eta = eta0 * tau_update / tauB
            plot_weights(dend_mon, events, bursts, spikes, [targetB, B0, eta, tauB, tau_update, smoothwin_], taskdir)
            plot_rasters(spksSE, bursts, targetB, isis, runtime_, taskdir)
        else:
            # calculate neurometric per neuron
            events, bursts, singles, spikes, isis = spks2neurometric(spksSE, runtime, settletime, validburst,
                                                                     smoothwin=smoothwin_, raster=True)
            plot_neurometric(events, bursts, spikes, stim1, stim2, stimtime,
                             (settletime_, runtime_), taskdir, smoothwin_)
            plot_isis(isis, bursts, events, (settletime_, runtime_), taskdir)

    # -------------------------------------
    # Choice selection
    # -------------------------------------
    # population rates and downsample
    originaltime = rateDE1.t / second
    interptime = np.linspace(0, originaltime[-1], originaltime[-1]*100)     # every 10 ms
    fDE1 = interpolate.interp1d(originaltime, rateDE1.smooth_rate(window='flat', width=smoothwin))
    fDE2 = interpolate.interp1d(originaltime, rateDE2.smooth_rate(window='flat', width=smoothwin))
    fSE1 = interpolate.interp1d(originaltime, rateSE1.smooth_rate(window='flat', width=smoothwin))
    fSE2 = interpolate.interp1d(originaltime, rateSE2.smooth_rate(window='flat', width=smoothwin))
    rateDE = np.array([f(interptime) for f in [fDE1, fDE2]])
    rateSE = np.array([f(interptime) for f in [fSE1, fSE2]])

    # select the last half second of the stimulus
    newdt = runtime_ / rateDE.shape[1]
    settletimeidx = int(settletime_ / newdt)
    dec_ival = np.array([(stimoff_ - 0.5) / newdt, stimoff_ / newdt], dtype=int)
    who_wins = rateDE[:, dec_ival[0]:dec_ival[1]].mean(axis=1)

    # divide trls into preferred and non-preferred
    pref_msk = np.argmax(who_wins)
    poprates_dec = np.array([rateDE[pref_msk], rateDE[~pref_msk]])  # 0: pref, 1: npref
    poprates_sen = np.array([rateSE[pref_msk], rateSE[~pref_msk]])

    results = {
        'raw_data': {'poprates_dec': poprates_dec[:, settletimeidx:],
                     'poprates_sen': poprates_sen[:, settletimeidx:],
                     'pref_msk': np.array([pref_msk]),
                     'last_muOUd': last_muOUd},
        'sim_state': np.zeros(1),
        'computed': {'events': events, 'bursts': bursts, 'singles': singles, 'spikes': spikes, 'isis': np.array(isis)}}

    return results


class JobInfoExperiment(Experiment):
    run_task = staticmethod(run_task_hierarchical)

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

        param_fixed = {
            'dec': {
                'populations': {
                    'N_E': Parameter(16),
                    'N_I': Parameter(4),
                    'sub': Parameter(0.15)},
                'connectivity': {
                    'w_p': Parameter(1.6),
                    'gEEa': Parameter(0.05, 'nS'),
                    'gEEn': Parameter(0.165, 'nS'),
                    'gEIa': Parameter(0.04, 'nS'),
                    'gEIn': Parameter(0.13, 'nS'),
                    'gIE': Parameter(1.33, 'nS'),
                    'gII': Parameter(1.0, 'nS'),
                    'delay': Parameter(0.5, 'ms'),
                    'gXE': Parameter(2.1, 'nS'),
                    'gXI': Parameter(1.62, 'nS')},
                'neuron': {
                    'CmE': Parameter(500, 'pF'),
                    'CmI': Parameter(250, 'pF'),
                    'gleakE': Parameter(25, 'nS'),
                    'gleakI': Parameter(20, 'nS'),
                    'Vl': Parameter(-70, 'mV'),
                    'Vt': Parameter(-50, 'mV'),
                    'Vr': Parameter(-55, 'mV'),
                    'tau_refE': Parameter(2, 'ms'),
                    'tau_refI': Parameter(1, 'ms'),
                    'nu_ext': Parameter(2400, 'Hz'),
                    'nu_ext1': Parameter(2392, 'Hz')},
                'synapse': {
                    'VrevE': Parameter(0, 'mV'),
                    'VrevI': Parameter(-70, 'mV'),
                    'tau_ampa': Parameter(2, 'ms'),
                    'tau_gaba': Parameter(5, 'ms'),
                    'tau_nmda_d': Parameter(100, 'ms'),
                    'tau_nmda_r': Parameter(2, 'ms'),
                    'alpha_nmda': Parameter(0.5, 'kHz')}},

            'sen': {
                'populations': {
                    'N_E': Parameter(1600),
                    'N_I': Parameter(400),
                    'N_X': Parameter(1000),
                    'sub': Parameter(0.5)},
                'connectivity': {
                    'eps': Parameter(0.2),
                    'w_p': Parameter(1.3),
                    'gEE': Parameter(0.7589, 'nS'),
                    'gEI': Parameter(1.5179, 'nS'),
                    'gIE': Parameter(12.6491, 'nS'),
                    'gII': Parameter(12.6491, 'nS'),
                    'gmax': Parameter(100),
                    'epsX': Parameter(0.32),
                    'alphaX': Parameter(0),
                    'gXE': Parameter(1.7076, 'nS'),
                    'gXI': Parameter(1.7076, 'nS')},
                'neuron': {
                    'CmE': Parameter(250, 'pF'),
                    'CmI': Parameter(250, 'pF'),
                    'gleakE': Parameter(16.7, 'nS'),
                    'gleakI': Parameter(16.7, 'nS'),
                    'Vl': Parameter(-70, 'mV'),
                    'Vt': Parameter(-50, 'mV'),
                    'Vr': Parameter(-60, 'mV'),
                    'tau_refE': Parameter(2, 'ms'),
                    'tau_refI': Parameter(1, 'ms'),
                    'nu_ext': Parameter(12.5, 'Hz')},
                '2c': {
                    'Cms': Parameter(370, 'pF'),
                    'taus': Parameter(16, 'ms'),
                    'tauws': Parameter(100, 'ms'),
                    'bws': Parameter(-200, 'pA'),
                    'gCas': Parameter(1300, 'pA'),
                    'tau_refE': Parameter(3, 'ms'),
                    'gEEs': Parameter(1.051, 'nS'),
                    'gIEs': Parameter(17.515, 'nS'),
                    'gXEs': Parameter(2.365, 'nS'),
                    'Cmd': Parameter(170, 'pF'),
                    'taud': Parameter(7, 'ms'),
                    'tauwd': Parameter(30, 'ms'),
                    'awd': Parameter(-13, 'nS'),
                    'gCad': Parameter(1200, 'pA'),
                    'bpA': Parameter(2600, 'pA'),
                    'k1': Parameter(0.5, 'ms'),
                    'k2': Parameter(2.5, 'ms'),
                    'tauOU': Parameter(2, 'ms'),
                    'sigmaOU': Parameter(450, 'pA'),
                    'muOUs': Parameter(70, 'pA'),
                    'muOUd': Parameter(-270, 'pA'),
                    # 'targetB': Parameter(2, 'Hz'),
                    'tauB': Parameter(50000, 'ms'),
                    'tau_update': Parameter(10, 'ms'),
                    'eta0': Parameter(5, 'pA'),
                    'validburst': Parameter(16e-3),  # in seconds
                    'min_burst_stop': Parameter(0.1)},
                'synapse': {
                    'VrevE': Parameter(0, 'mV'),
                    'VrevI': Parameter(-80, 'mV'),
                    'tau_decay': Parameter(5, 'ms'),
                    'tau_rise': Parameter(1, 'ms'),
                    'VrevIsd': Parameter(-70, 'mV'),
                    'tau_decaysd': Parameter(1, 'ms')}},

            'simulation': {
                'seedcon': Parameter(1284),
                'runtime': Parameter(1000, 'second'),
                'settletime': Parameter(0, 'second'),
                'stimon': Parameter(0, 'second'),
                'stimoff': Parameter(1000, 'second'),
                'smoothwin': Parameter(100, 'ms'),
                '2cmodel': True,
                'pltfig1': True,
                'burstanalysis': True,
                'plasticdend': True,            # needs to be run with pltfig1 True and 2cmodel True!
                'nummethod': 'euler'},

            'stimulus': {
                'I0': Parameter(80, 'pA'),
                'I0s': Parameter(120, 'pA'),
                'mu1': Parameter(0.25),
                'mu2': Parameter(-0.25),
                'sigma': Parameter(1),
                'taustim': Parameter(20, 'ms'),
                'replicate': False}}

        self.tables.add_parameters(param_fixed)

        param_ranges = {
            'c': ParameterArray(np.array([0])),
            'bfb': ParameterArray(np.array([0])),
            'targetB': ParameterArray(np.arange(1.5, 4.5, 0.5), 'Hz'),
            'iter': ParameterArray(np.arange(0, 4))
        }
        # 'bfb': ParameterArray(np.arange(0, 7)),
        # 'iter': ParameterArray(np.arange(0, 10)),

        self.tables.add_parameter_ranges(param_ranges)

        # One to one link between parameters, avoids unnecessary combinations
        # self.tables.link_parameter_ranges([('tau_update',), ('B0',)])


if __name__ == '__main__':
    from snep.parallel2 import run

    '''
        IMPORTANT: Only include code here that can be run repeatedly,
        because this will be run once in the parent process, and then
        once for every worker process.
        '''
    # the path which is expanded here, can be different from result_dir
    ji_kwargs = dict(root_dir=os.path.expanduser('~/Documents/WS19/MasterThesis/Experiments'))
    job_info = run(JobInfoExperiment, ji_kwargs, username=username, max_tasks=max_tasks, mem_per_task=mem_per_task,
                   max_task_time=max_task_time, poll_interval=poll_interval,
                   result_dir='Documents/WS19/MasterThesis/Experiments',
                   additional_files=['integration_circuit.py', 'sensory_circuit.py', 'burstQuant.py'])
