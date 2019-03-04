import numpy as np
from scipy.signal import lfilter


def adjust_variable(var_prev, cm_prev, cm_new):
    """
    Adjusts conductances or current for different cell capacitances.
    """
    target = var_prev / cm_prev
    return target * cm_new


def unitless(quantity, time_unit):
    """
    Removes units from the quantity
    """
    return int(quantity / time_unit)


def get_OUstim(n, tau):
    """
    Ornstein-Uhlenbeck process in discrete time
    """
    a = np.exp(-(1 / tau))
    i = lfilter(np.ones(1), [1, -a], np.sqrt(1 - a * a) * np.random.randn(n))
    return np.asanyarray(i)


# def plasticity():
#     eta0 = task_info['sen']['2c']['eta0']
#     tauB = task_info['sen']['2c']['tauB']
#     targetB = task_info['targetB']
#     B0 = tauB*targetB
#     tau_update = task_info['sen']['2c']['tau_update']
#     eta = eta0 * tau_update / tauB
#     validburst = task_info['sen']['2c']['validburst']      # if tau_burst = 4*ms, at 16 ms stopburst = 0.0183
#     min_burst_stop = task_info['sen']['2c']['min_burst_stop']   # thus, we will set it at critrefburst = 0.02
#     tau_burst = -validburst / np_log(min_burst_stop) * second
#
#
#
# # plotting functions
# def smoothpoprate(raster, sub, win=50e-3, dt=1e-3):
#     """
#
#     :param raster: a 2D array containig spiketimes of spikes, events or bursts
#     :param sub: index that represents end of pop1 and beginnig of pop2
#     :param dt:  integration time step
#     :param win: width of kernel window
#     :return: the smoothed rate of pop1 and pop2
#     """
#     # population rates by averaging and binning
#     popraster1 = np.squeeze(raster[:sub].mean(axis=0)) / win
#     popraster2 = np.squeeze(raster[sub:].mean(axis=0)) / win
#
#     # smooth rate
#     kernel = np.ones((int(win / dt)))
#     rate1 = np.convolve(popraster1, kernel, mode='same')
#     rate2 = np.convolve(popraster2, kernel, mode='same')
#
#     # variables2return
#     rate = np.vstack((rate1, rate2))
#     sub = int(rate.shape[0] /2)
#
#     return rate, sub
#
# def plot_neurometric(events, bursts, spikes, stim1, stim2, stimtime, time01, taskdir, smooth_win=50e-3):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#
#     # params from dimensions
#     subSE = int(events.shape[0] / 2)
#     time = np.linspace(time01[0], time01[1], events.shape[1])
#
#     # obtain rates if events, bursts are rasters
#     if subSE > 1:
#         spikes, _ = smoothpoprate(spikes, subSE, win=smooth_win)
#         events, _ = smoothpoprate(events, subSE, win=smooth_win)
#         bursts, subSE = smoothpoprate(bursts, subSE, win=smooth_win)
#
#     # start figure
#     fig2, axs = plt.subplots(5, 2, figsize=(16, 10), sharex='row', sharey='row')
#
#     # stimulus
#     axs[0, 0].plot(stimtime, stim1.mean(axis=0)*1e12, color='C3', lw=1.5)
#     axs[0, 0].set_title('sensory population 1')
#     axs[0, 0].set_ylabel(r'$som_{input}$ (pA)')
#     axs[0, 0].set_xlim(time[0]-0.2, time[-1]+0.2)
#     axs[0, 1].plot(stimtime, stim2.mean(axis=0)*1e12, color='C0', lw=1.5)
#     axs[0, 1].set_title('sensory population 2')
#
#     # firing rate
#     axs[1, 0].plot(time, spikes[:subSE].mean(axis=0), lw=1.5, color='C5')
#     axs[1, 0].set_ylabel(r'$A$ (Hz)')
#     axs[1, 1].plot(time, spikes[subSE:].mean(axis=0), lw=1.5, color='C5')
#     # plt.ylim(0, 25)
#
#     # burst rate
#     axs[2, 0].plot(time, bursts[:subSE].mean(axis=0), lw=1.5, color='C1')
#     axs[2, 0].set_ylabel(r'$B$ (Hz)')
#     axs[2, 1].plot(time, bursts[subSE:].mean(axis=0), lw=1.5, color='C1')
#
#     # burst fraction
#     bfracc1 = bursts[:subSE].mean(axis=0) / events[:subSE].mean(axis=0)
#     bfracc1[np.isnan(bfracc1)] = 0  # handle division by zero
#     bfracc2 = bursts[subSE:].mean(axis=0) / events[subSE:].mean(axis=0)
#     bfracc2[np.isnan(bfracc2)] = 0  # handle division by zero
#     axs[3, 0].plot(time, bfracc1*100, lw=1.5, color='C6')
#     axs[3, 0].set_ylabel(r'$F$ (%)')
#     axs[3, 1].plot(time, bfracc2*100, lw=1.5, color='C6')
#
#     # event rate
#     axs[4, 0].plot(time, events[:subSE].mean(axis=0), lw=1.5, color='C2')
#     axs[4, 0].set_ylabel(r'$E$ (Hz)')
#     axs[4, 0].set_xlabel(r'$time$ (ms)')
#     axs[4, 1].plot(time, events[subSE:].mean(axis=0), lw=1.5, color='C2')
#     axs[4, 1].set_xlabel(r'$time$ (ms)')
#
#     fig2.savefig(taskdir + '/figure2.png')
#     plt.close(fig2)
#
#
# def plot_isis(isis, bursts, events, time01, taskdir, maxisi=500):
#     """
#     bursts and events are rasters (0 or 1)
#     :param isis:
#     :param bursts:
#     :param events:
#     :param time01:
#     :param taskdir:
#     :param maxisi:
#     :return:
#     """
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     sns.set(context='paper', style='darkgrid')
#
#     # params from dimensions
#     n, timepts = events.shape
#     time = np.linspace(time01[0], time01[1], timepts)
#     time = np.repeat(time[np.newaxis, :], n, axis=0)
#
#     # get different versions of isis
#     isis_b = np.diff(time[bursts.astype(bool)].reshape(n, -1), axis=1).flatten()
#     isis_e = np.diff(time[events.astype(bool)].reshape(n, -1), axis=1).flatten()
#     cv = isis.std() / isis.mean()
#     cv_b = isis_b.std() / isis_b.mean()
#     cv_e = isis_e.std() / isis_e.mean()
#
#     # start figure
#     fig3, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=False)
#
#     # all isis
#     sns.distplot(isis, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C5', ax=axs[0])
#     axs[0].plot(0, c='white', label=r'CV = %.3f' % cv)
#     axs[0].legend(loc='upper right')
#     axs[0].set_ylabel(r'$Proportion$ $of$ $isi$')
#
#     # burst isis
#     sns.distplot(isis_b, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C1', ax=axs[1])
#     axs[1].plot(0, c='white', label=r'CV_b = %.3f' % cv_b)
#     axs[1].legend(loc='upper right')
#     axs[1].set_title('Interspike interval distribution')
#     axs[1].set_xlabel(r'$Interspike$ $interval$ (ms)')
#     axs[1].set_xlim(0, maxisi)
#
#     # event isis
#     sns.distplot(isis_e, kde=False, norm_hist=True, bins=np.linspace(0, maxisi, 101), color='C2', ax=axs[2])
#     axs[2].plot(0, c='white', label=r'CV_e = %.3f' % cv_e)
#     axs[2].legend(loc='upper right')
#
#     # save figure
#     plt.tight_layout()
#     fig3.savefig(taskdir + '/figure3.png')
#     plt.close(fig3)
#
#
# def create_inset(axes, data2plt, c, xlim, w=1, h=0.7, nyticks=4):
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#
#     axins = inset_axes(axes, w, h, loc=1)
#     axins.plot(data2plt[0], data2plt[1], color=c, lw=1.5)
#     axins.set_xlim(xlim)
#     axins.yaxis.get_major_locator().set_params(nbins=nyticks)
#     plt.xticks(visible=False)
#
#     return axins
#
#
# def plot_weights(mon, events, bursts, spikes, infosimul, taskdir):
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from brian2.units import Hz, pA, ms
#     sns.set(context='paper', style='ticks')
#
#     # params
#     target, B0, eta, tauB, tau_update, smooth_win = infosimul
#     subSE = int(events.shape[0] / 2)
#     time = np.linspace(0, mon.t_[-1], bursts.shape[1])
#     lasttime = time[-1]
#     zoomt = (lasttime-1, lasttime)
#     dt = lasttime / bursts.shape[1]
#     target /= Hz
#     eta /= pA
#     tauB /= ms
#     tau_update /= ms
#     nn2plt = 10
#
#     fig2, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
#     fig2.add_axes(axs[0, 0])
#     plt.title(r'Drift of dendritic noise ($\mu_{OU_{d}}$)')
#     plt.plot(mon.t_, mon.muOUd[:nn2plt].T*1e12, color='gray', lw=0.5)
#     plt.plot(mon.t_, mon.muOUd.mean(axis=0)*1e12, color='C6', lw=1.5)
#     plt.ylabel(r'$weights$ $(pA)$')
#     create_inset(axs[0, 0], (mon.t_, mon.muOUd.mean(axis=0)*1e12), 'C6', zoomt)
#
#     fig2.add_axes(axs[1, 0])
#     plt.title(r'Noise background current to dendrites ($I_{OU_{d}}$)')
#     plt.plot(mon.t_, mon.Ibg[:nn2plt].T*1e9, color='gray', lw=0.5)
#     plt.plot(mon.t_, mon.Ibg.mean(axis=0)*1e9, color='C0', lw=1.5)
#     plt.ylabel('$I_{OU_{d}}$ $(nA)$')
#     create_inset(axs[1, 0], (mon.t_, mon.Ibg.mean(axis=0)*1e9), 'C0', zoomt)
#
#     fig2.add_axes(axs[2, 0])
#     plt.title(r'Excitation current from integration circuit')
#     plt.plot(mon.t_, mon.g_ea[:nn2plt].T, color='gray', lw=0.5)
#     plt.plot(mon.t_, mon.g_ea.mean(axis=0), color='C3', lw=1.5)
#     plt.xlabel(r'$Time$ (s)')
#     plt.ylabel(r'$g_{AMPA}$ $(~nS)$')
#     create_inset(axs[2, 0], (mon.t_, mon.g_ea.mean(axis=0)), 'C3', zoomt)
#
#     fig2.add_axes(axs[0, 1])
#     plt.title('PSTH')
#     plt.plot(time, spikes[:subSE].mean(axis=0), lw=1.5, color='C5')
#     plt.ylabel(r'$Firing$ $rate$ $(Hz)$')
#     create_inset(axs[0, 1], (time, spikes[:subSE].mean(axis=0)), 'C5', zoomt)
#
#     fig2.add_axes(axs[1, 1])
#     #plt.title('Burst rate')
#     plt.plot(time, bursts[:subSE].mean(axis=0), lw=1.5, color='C1')
#     plt.axhline(target, color='gray', lw=2, ls='dashed')
#     plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
#     #plt.ylim(-0.5, 5.5)
#     create_inset(axs[1, 1], (time, bursts[:subSE].mean(axis=0)), 'C1', zoomt)
#
#     fig2.add_axes(axs[2, 1])
#     bfracc = bursts[:subSE].mean(axis=0) / events[:subSE].mean(axis=0)
#     bfracc[np.isnan(bfracc)] = 0    # handle division by 0 - because no event also means no burst!
#     # epsilon = 1e-5    # +epsilon avoids division by 0, but it's not needed here.
#     #plt.title('Burst fraction')
#     plt.plot(time, bfracc*100, lw=1.5, color='C2')
#     plt.ylabel(r'$Burst$ $fraction$ $(\%)$')
#     plt.xlabel(r'$Time$ $(s)$')
#     plt.ylim(0, 100)
#     create_inset(axs[2, 1], (time, bfracc*100), 'C2', zoomt)
#
#     fig2.add_axes(axs[0, 2])
#     B = mon.B.mean(axis=0)
#     plt.plot(mon.t_, mon.B[:nn2plt].T - B0, color='gray', lw=0.5)
#     plt.plot(mon.t_, B - B0, color='C4', lw=1.5)
#     plt.ylabel(r'$B - B0$')
#     create_inset(axs[0, 2], (mon.t_, B-B0), 'C4', zoomt)
#
#     fig2.add_axes(axs[1, 2])
#     plt.title('Integral of B in a 10 sec span')
#     if lasttime > 50:
#         a1_start = int(10/dt)               # from 10:20 sec
#         a2_start = int((time[-1]/2)/dt)       # middle 10 sec
#         a3_start = int((time[-1]-10)/dt)    # last 10 sec
#         step = a1_start
#         area1 = B[a1_start:a1_start + step].sum() * dt
#         area2 = B[a2_start:a2_start + step].sum() * dt
#         area3 = B[a3_start:a3_start + step].sum() * dt
#
#         plt.plot(time[a1_start], np.abs(area1), 'o', label='initial')
#         plt.plot(time[a2_start], np.abs(area2), 'o', label='middle')
#         plt.plot(time[a3_start], np.abs(area3), 'o', label='late')
#         plt.legend(loc='best')
#     plt.ylabel(r'$AUC$ $(au)$')
#
#     fig2.add_axes(axs[2, 2])
#     plt.title('Info of simulation')
#     plt.plot(0, c='white', label=r'target = %.1f Hz' % target)
#     plt.plot(0, c='white', label=r'eta = %.3f pA' % eta)
#     plt.plot(0, c='white', label=r'tauB = %i ms' % tauB)
#     plt.plot(0, c='white', label=r'tau_update = %i ms' % tau_update)
#     plt.legend(loc='best', fontsize='large', frameon=False)
#     plt.xlabel(r'$Time$ $(s)$')
#
#     #plt.tight_layout()
#     fig2.savefig(taskdir + '/figure2.png')
#     plt.close(fig2)
#
#
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
#
#
#
#
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
