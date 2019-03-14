import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import lfilter
from brian2.units import second, Hz, pA, ms
from brian2tools import plot_raster, plot_rate


def adjust_variable(var_prev, cm_prev, cm_new):
    """
    Adjusts conductances or current for different cell capacitances.
    """
    target = var_prev / cm_prev
    return target * cm_new


def unitless(quantity, time_unit, as_int=True):
    """
    Removes units from the quantity
    """
    if as_int:
        return int(quantity / time_unit)
    return  quantity / time_unit


def get_OUstim(n, tau):
    """
    Ornstein-Uhlenbeck process in discrete time
    """
    a = np.exp(-(1 / tau))
    i = lfilter(np.ones(1), [1, -a], np.sqrt(1 - a * a) * np.random.randn(n))
    return np.asanyarray(i)


def handle_downsampled_spks(spk_times):
    """transforms 2 spks in one dt to 2 consecutive spks"""
    conflict = np.where(np.diff(spk_times) == 0)[0]
    spk_times[conflict] -= 1
    # recursive search for no 2 spks in one dt bin
    if np.any(np.diff(spk_times) == 0):
        spk_times = handle_downsampled_spks(spk_times)
    return spk_times


def plot_fig1(monitors, win, taskdir):
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


def plot_fig2(task_info, events, bursts, spikes, stim1, stim2, stim_time, taskdir):
    # params
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    time = np.linspace(settle_time, runtime, events.shape[1])
    subSE = int(events.shape[0] / 2)

    # start figure
    fig2, axs = plt.subplots(5, 2, figsize=(16, 10), sharex='row', sharey='row')

    # stimulus
    axs[0, 0].plot(stim_time, stim1.mean(axis=0) * 1e12, color='C3', lw=1.5)
    axs[0, 0].set_title('sensory population 1')
    axs[0, 0].set_ylabel(r'$som_{input}$ (pA)')
    axs[0, 0].set_xlim(time[0]-0.2, time[-1]+0.2)
    axs[0, 1].plot(stim_time, stim2.mean(axis=0) * 1e12, color='C0', lw=1.5)
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


def plot_fig3(task_info, dend_mon, events, bursts, spikes, taskdir):
    sns.set(context='paper', style='ticks')

    # params
    dt = task_info['sim']['stim_dt']
    eta0 = unitless(task_info['plastic']['eta0'], pA)
    tauB = unitless(task_info['plastic']['tauB'], ms)
    tau_update = unitless(task_info['plastic']['tau_update'], ms)
    target = unitless(task_info['targetB'], Hz, as_int=False)
    B0 = tauB * target
    eta = eta0 * tau_update / tauB
    subSE = int(events.shape[0] / 2)
    time = np.linspace(0, dend_mon.t_[-1], bursts.shape[1])
    last_time = time[-1]
    zoom_inteval = (last_time-1, last_time)
    nn2plt = 10

    fig3, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig3.add_axes(axs[0, 0])
    plt.title(r'Drift of dendritic noise ($\mu_{OU_{d}}$)')
    plt.plot(dend_mon.t_, dend_mon.muOUd[:nn2plt].T * 1e12, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, dend_mon.muOUd.mean(axis=0) * 1e12, color='C6', lw=1.5)
    plt.ylabel(r'$weights$ $(pA)$')
    create_inset(axs[0, 0], (dend_mon.t_, dend_mon.muOUd.mean(axis=0) * 1e12), 'C6', zoom_inteval)

    fig3.add_axes(axs[1, 0])
    plt.title(r'Noise background current to dendrites ($I_{OU_{d}}$)')
    plt.plot(dend_mon.t_, dend_mon.Ibg[:nn2plt].T * 1e9, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, dend_mon.Ibg.mean(axis=0) * 1e9, color='C0', lw=1.5)
    plt.ylabel('$I_{OU_{d}}$ $(nA)$')
    create_inset(axs[1, 0], (dend_mon.t_, dend_mon.Ibg.mean(axis=0) * 1e9), 'C0', zoom_inteval)

    fig3.add_axes(axs[2, 0])
    plt.title(r'Excitation current from integration circuit')
    plt.plot(dend_mon.t_, dend_mon.g_ea[:nn2plt].T, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, dend_mon.g_ea.mean(axis=0), color='C3', lw=1.5)
    plt.xlabel(r'$Time$ (s)')
    plt.ylabel(r'$g_{AMPA}$ $(~nS)$')
    create_inset(axs[2, 0], (dend_mon.t_, dend_mon.g_ea.mean(axis=0)), 'C3', zoom_inteval)

    fig3.add_axes(axs[0, 1])
    plt.title('PSTH')
    plt.plot(time, spikes[:subSE].mean(axis=0), lw=1.5, color='C5')
    plt.ylabel(r'$Firing$ $rate$ $(Hz)$')
    create_inset(axs[0, 1], (time, spikes[:subSE].mean(axis=0)), 'C5', zoom_inteval)

    fig3.add_axes(axs[1, 1])
    #plt.title('Burst rate')
    plt.plot(time, bursts[:subSE].mean(axis=0), lw=1.5, color='C1')
    plt.axhline(target, color='gray', lw=2, ls='dashed')
    plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
    #plt.ylim(-0.5, 5.5)
    create_inset(axs[1, 1], (time, bursts[:subSE].mean(axis=0)), 'C1', zoom_inteval)

    fig3.add_axes(axs[2, 1])
    bfracc = bursts[:subSE].mean(axis=0) / events[:subSE].mean(axis=0)
    bfracc[np.isnan(bfracc)] = 0    # handle division by 0 - because no event also means no burst!
    plt.plot(time, bfracc*100, lw=1.5, color='C2')
    plt.ylabel(r'$Burst$ $fraction$ $(\%)$')
    plt.xlabel(r'$Time$ $(s)$')
    plt.ylim(0, 100)
    create_inset(axs[2, 1], (time, bfracc*100), 'C2', zoom_inteval)

    fig3.add_axes(axs[0, 2])
    B = dend_mon.B.mean(axis=0)
    plt.plot(dend_mon.t_, dend_mon.B[:nn2plt].T - B0, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, B - B0, color='C4', lw=1.5)
    plt.ylabel(r'$B - B0$')
    create_inset(axs[0, 2], (dend_mon.t_, B - B0), 'C4', zoom_inteval)

    fig3.add_axes(axs[1, 2])
    if last_time > 50:
        # from 10:20 sec, middle 10 sec and last 10 sec
        starts = [int(10/dt), int((last_time/2)/dt), int((last_time-10)/dt)]
        areas = [B[start:start + starts[0]].sum() * dt for start in starts]

        for idx, label in enumerate(['initial', 'middle', 'late']):
            plt.plot(time[starts[idx]], np.abs(areas[idx]), 'o', label=label)
        plt.title('Integral of B in a 10 sec span')
        plt.ylabel(r'$AUC$ $(au)$')
        plt.legend(loc='best')

    fig3.add_axes(axs[2, 2])
    plt.title('Info of simulation')
    plt.plot(0, c='white', label=r'target = %.1f Hz' % target)
    plt.plot(0, c='white', label=r'eta = %.3f pA' % eta)
    plt.plot(0, c='white', label=r'tauB = %i ms' % tauB)
    plt.plot(0, c='white', label=r'tau_update = %i ms' % tau_update)
    plt.legend(loc='best', fontsize='large', frameon=False)
    plt.xlabel(r'$Time$ $(s)$')

    fig3.savefig(taskdir + '/figure3.png')
    plt.close(fig3)


def plot_plastic_rasters(task_info, spk_mon, bursts, isis, taskdir):
    sns.set(context='paper', style='darkgrid')

    # params
    target = unitless(task_info['targetB'], Hz, as_int=False)
    last_time = unitless(task_info['sim']['runtime'], second)
    subSE = min(int(spk_mon.source.__len__() / 2), 20)
    nticks = 4

    fig4, axs = plt.subplots(4, 1, figsize=(4, 12), sharex=False)

    # sensory circuit
    fig4.add_axes(axs[0])
    plt.title('Before plasticity')
    spksSE1 = spk_mon.i < subSE
    plot_raster(spk_mon.i[spksSE1], spk_mon.t[spksSE1], color='gray', marker='.', markersize=0.7, time_unit=second)
    plt.yticks(np.arange(0, subSE+1, subSE/nticks))
    plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
    plt.xlim(5, 10)
    plt.xlabel('')

    fig4.add_axes(axs[1])
    plt.title('After plasticity')
    plot_raster(spk_mon.i[spksSE1], spk_mon.t[spksSE1], color='C5', marker='.', markersize=0.7, time_unit=second)
    plt.yticks(np.arange(0, subSE+1, subSE/nticks))
    plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
    plt.xlim(last_time - 5, last_time)
    plt.xlabel(r'$Times$ $(s)$')

    fig4.add_axes(axs[2])
    maxbrate = 5.5
    plt.plot(target, bursts[:subSE].mean(axis=0)[int(last_time - 10):].mean(), c='C1', marker='o')
    plt.plot(np.linspace(0, maxbrate, 100), np.linspace(0, maxbrate, 100), c='gray', lw=1.5, ls='dashed')
    plt.xlabel(r'$Target$ $rate$ $(Hz)$')
    plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
    plt.xlim(0, 5.5)
    plt.ylim(0, 5.5)

    fig4.add_axes(axs[3])
    maxisi = 500
    isis = isis[isis > 0]
    cv = isis.std() / isis.mean()
    sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2')
    plt.plot(0, c='white', label=r'CV = %.3f' % cv)
    plt.title('Interspike interval distribution')
    plt.xlabel(r'$Interspike$ $interval$ $(ms)$')
    plt.ylabel(r'$Percent$ $of$ $isi$')
    plt.xlim(0, maxisi)
    plt.legend(loc='best')

    plt.tight_layout()
    fig4.savefig(taskdir + '/figure4.png')
    plt.close(fig4)


def plot_isis(task_info, isis, burst_array, event_array, taskdir, maxisi=500):
    sns.set(context='paper', style='darkgrid')

    # params
    nn, timepts = event_array.shape
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    time = np.linspace(settle_time, runtime, timepts)

    # get different versions of isis
    ibis = np.zeros(1)
    ieis = ibis.copy()
    for n in np.arange(nn):
        ibi = np.diff(time[burst_array[n].astype(bool)])
        iei = np.diff(time[event_array[n].astype(bool)])
        ibis = np.hstack((np.zeros(1), ibi[ibi > 0], ibis))
        ieis = np.hstack((np.zeros(1), iei[iei > 0], ieis))
    # remove zero inter-intervals
    for isis_ in [isis, ibis, ieis]:
        isis_ = isis_[isis_ > 0]
    cv = isis.std() / isis.mean()
    cv_b = ibis.std() / ibis.mean()
    cv_e = ieis.std() / ieis.mean()

    # start figure
    fig5, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=False)

    # all isis
    sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C5', ax=axs[0])
    axs[0].plot(0, c='white', label=r'CV = %.3f' % cv)
    axs[0].legend(loc='upper right')
    axs[0].set_ylabel(r'$Proportion$ $of$ $isi$')

    # burst isis
    sns.distplot(ibis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C1', ax=axs[1])
    axs[1].plot(0, c='white', label=r'CV_b = %.3f' % cv_b)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Interspike interval distribution')
    axs[1].set_xlabel(r'$Interspike$ $interval$ (ms)')
    axs[1].set_xlim(0, maxisi)

    # event isis
    sns.distplot(ieis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2', ax=axs[2])
    axs[2].plot(0, c='white', label=r'CV_e = %.3f' % cv_e)
    axs[2].legend(loc='upper right')

    # save figure
    plt.tight_layout()
    fig5.savefig(taskdir + '/figure5.png')
    plt.close(fig5)


def create_inset(axes, data2plt, c, xlim, w=1, h=0.7, nyticks=4):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(axes, w, h, loc=1)
    axins.plot(data2plt[0], data2plt[1], color=c, lw=1.5)
    axins.set_xlim(xlim)
    axins.yaxis.get_major_locator().set_params(nbins=nyticks)
    plt.xticks(visible=False)

    return axins


# def plot_rasters(spkmon, bursts, target, isis, lasttime, taskdir):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from brian2.units import second
#     from brian2tools import plot_raster
#     sns.set(context='paper', style='darkgrid')
#
#     # params
#     subSE = min(int(spkmon.source.__len__() / 2), 20)
#     nticks = 4
#
#     fig3, axs = plt.subplots(4, 1, figsize=(4, 12), sharex=False)
#
#     # sensory circuit
#     fig3.add_axes(axs[0])
#     plt.title('Before plasticity')
#     spksSE1 = spkmon.i < subSE
#     plot_raster(spkmon.i[spksSE1], spkmon.t[spksSE1], color='gray', marker='.', markersize=0.7, time_unit=second)
#     plt.yticks(np.arange(0, subSE+1, subSE/nticks))
#     plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
#     plt.xlim(5, 10)
#     plt.xlabel('')
#
#     fig3.add_axes(axs[1])
#     plt.title('After plasticity')
#     plot_raster(spkmon.i[spksSE1], spkmon.t[spksSE1], color='C5', marker='.', markersize=0.7, time_unit=second)
#     plt.yticks(np.arange(0, subSE+1, subSE/nticks))
#     plt.ylabel(r'$neuron$ $index$')  # , {'horizontalalignment':'right'})
#     plt.xlim(lasttime-5, lasttime)
#     plt.xlabel(r'$Times$ $(s)$')
#
#     fig3.add_axes(axs[2])
#     maxbrate = 5.5
#     plt.plot(target, bursts[:subSE].mean(axis=0)[int(lasttime-10):].mean(), c='C1', marker='o')
#     plt.plot(np.linspace(0, maxbrate, 100), np.linspace(0, maxbrate, 100), c='gray', lw=1.5, ls='dashed')
#     plt.xlabel(r'$Target$ $rate$ $(Hz)$')
#     plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
#     plt.xlim(0, 5.5)
#     plt.ylim(0, 5.5)
#
#     fig3.add_axes(axs[3])
#     maxisi = 500
#     cv = isis.std() / isis.mean()
#     sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2')
#     plt.plot(0, c='white', label=r'CV = %.3f' % cv)
#     plt.title('Interspike interval distribution')
#     plt.xlabel(r'$Interspike$ $interval$ $(ms)$')
#     plt.ylabel(r'$Percent$ $of$ $isi$')
#     plt.xlim(0, maxisi)
#     plt.legend(loc='best')
#
#     plt.tight_layout()
#     fig3.savefig(taskdir + '/figure3.png')
#     plt.close(fig3)
#
#
# # ideas to import params
#
#
# def create_filenames(circuit, extension='.pkl'):
#     """
#     Create the array of filenames which will serve as pickle objects
#     :param circuit: a str, 'sen' or 'dec'
#     :param extension: a str, file extension to use, default pickle object
#     :return: array of filenames
#     """
#     if isinstance(circuit, str) and isinstance(extension, str):
#         param_groups = ['pop_', 'recurrent_', 'external_', 'neurons_', 'synapses_']
#         return [s + circuit + extension for s in param_groups]
#
#
# def create_param_object(filename, variables):
#     """
#     Creates a pickle object where specific set of params are stored.
#     :param filename: a str, filename.pkl
#     :param variables: a list, contains the variables to dump
#     :return:
#     """
#     if validate_file(filename, ext='.pkl') and bool(variables):
#         with open('params/'.join(filename), 'wb') as pkl_file:
#             pickle.dump(variables, pkl_file)
#     return
#
#
# def load_param_object(filename, var_names):
#     """
#
#     :param filename:
#     :param var_names:
#     :return:
#     """
#     if validate_file(filename, ext='.pkl'):
#         pkl_file = open('params/'.join(filename), 'rb')
#         variables = pickle.load(pkl_file)
#         return variables
#
#
# def get_params(which_circuit):
#     """
#     Contains the all the params to construct different versions of hierarchical net.
#     It reads a particular subset and creates .pkl objects for the functions mk_*circuit to access these params.
#     :param which_circuit: a str, 'sen' or 'dec'
#     :return:
#     """
#     filenames = create_filenames(which_circuit)
#
#     # construct list of variables according to circuit
#     if which_circuit == 'sen':
#         pop_vars = [N_E, N_I, N_X, sub]
#         recurrent_vars = [eps, w_p, w_m, gEE, gEI, gIE, gII, gmax]
#         external_vars = [epsX, alphaX, gXE, gXI, nu_ext]
#         neurons_vars = [CmE, CmI, gleakE, gleakI, Vl, Vt, Vr, tau_refE, tau_refI]
#         synapse_vars = [VrevE, VrevI, tau_decay, tau_rise]
#         grouped_vars = [pop_vars, recurrent_vars, external_vars, neurons_vars, synapse_vars]
#
#     elif which_circuit == 'dec':
#         pop_vars = [N_E, N_I, sub]
#         recurrent_vars = [w_p, w_m, gEEa, gEEn, gEIa, gEIn, gIE, gII]
#         external_vars = [gXE, gXI, nu_ext, nu_ext1]
#         neurons_vars = [CmE, CmI, gleakE, gleakI, Vl, Vt, Vr, tau_refE, tau_refI]
#         synapse_vars = [VrevE, VrevI, tau_ampa, tau_gaba, tau_nmda_d, tau_nmda_r, alpha_nmda]
#         grouped_vars = [pop_vars, recurrent_vars, external_vars, neurons_vars, synapse_vars]
#
#     for file, this_vars in zip(filenames, grouped_vars):
#         if len(filenames) == len(grouped_vars):
#             create_param_object(file, this_vars)


def handle_brian_units(value):
    """
    Converts str values to floats, units.

    :param value: a str containing Brian2 supported values: 80*pA
    :return: a tuple with num_value and units if given any
    """
    num_units = value.split('*')
    try:
        num = float(num_units[0])
    except ValueError:
        # value is not a number (maybe a bool?) --> we don't care
        return None
    else:
        if len(num_units) > 1:
            units = num_units[1]
            return num, units
        else:
            return num


def brian2param(line):
    """
    Transforms a line containing an equivalence into valid key, value for params_dict.

    :param line: a str with the equivalence between variable and value + units, written in Brian2 style
    :return: key - a string
             value - a tuple containing a num and units, if any
    """
    split_line = line.split()
    # assumes equivalences are separated by white_space, ignore other in-line comments
    if '=' in split_line[1]:
        key = split_line[0]
        value = split_line[2]
        value = handle_brian_units(value)
        return key, value
    else:
        return


def validate_file(filename, extension='.txt'):
    from os import path
    if not isinstance(filename, str):
        raise ValueError("filename must be a string")

    if not filename.endswith(extension):
        print("Not a valid file. Input {} must be a .txt file.".format(filename))
        return False
    elif not path.exists(filename):
        print("Not a valid file. Path {} does not exists.".format(filename))
        return False
    else:
        return True


def read_params(filename='get_params.py'):
    """
    Reads params from a text file to a dictionary supported by snep.

    :param filename: a string, the path/filename to the params
    :return: dictionary with params, where values are instances of Parameter class
    """
    from snep.utils import Parameter

    params_default = {}
    if validate_file(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                # ignore commented lines
                if not line.lstrip().startswith('#'):
                    key, value = brian2param(line)
                    params_default[key] = Parameter(*value)
    return params_default


# maybe important, but not yet
def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        raise ValueError("{} not a bool".format(s))
