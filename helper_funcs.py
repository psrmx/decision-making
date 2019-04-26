import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.signal import lfilter
from brian2.units import second, Hz, pA, ms
from brian2tools import plot_raster, plot_rate
from sklearn import metrics as mtr


def adjust_variable(var_prev, cm_prev, cm_new):
    """Adjusts conductances or current for different cell capacitances"""
    target = var_prev / cm_prev
    return target * cm_new


def unitless(quantity, time_unit, as_int=True):
    """Removes units from the quantity"""
    if as_int:
        return int(quantity / time_unit)
    return quantity / time_unit


def get_OUstim(n, tau):
    """Ornstein-Uhlenbeck process in discrete time"""
    a = np.exp(-(1 / tau))
    i = lfilter(np.ones(1), [1, -a], np.sqrt(1 - a**2) * np.random.randn(n))
    return np.asanyarray(i)


def get_this_dt(task_params, tps):
    runtime = unitless(task_params['sim']['runtime'], second, as_int=False)
    settle_time = unitless(task_params['sim']['settle_time'], second, as_int=False)
    return np.round((runtime - settle_time)/tps, decimals=4)


def get_this_time(task_params, tps):
    settle_time = unitless(task_params['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_params['sim']['runtime'], second, as_int=False)
    return np.linspace(0, (runtime-settle_time), tps)


def handle_downsampled_spikes(spk_times):
    """transforms 2 spks in one dt to 2 consecutive spikes"""
    conflict = np.where(np.diff(spk_times) == 0)[0]
    spk_times[conflict] -= 1
    if np.any(np.diff(spk_times) == 0): # recursive search for no 2 spikes in one dt bin
        spk_times = handle_downsampled_spikes(spk_times)

    return spk_times


def instantaneous_rate(task_info, spikes):
    """computes the instantaneous spike count for a neuron at each trl"""
    n_trials, tps = spikes.shape
    new_dt = get_this_dt(task_info, tps)
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
    rates = np.empty((n_trials, tps))
    for i in range(n_trials):
        rates[i] = smooth_rate(spikes[i], smooth_win, new_dt, sub=[])[0]

    return rates


def smooth_rate(rate, smooth_win, dt, sub):
    """rectangular sliding window on a firing rate to smooth"""
    kernel = np.ones(int(smooth_win / dt))
    if bool(sub):
        rate1 = rate[:sub].mean(axis=0) / smooth_win
        rate2 = rate[sub:].mean(axis=0) / smooth_win
        smoothed_rate1 = np.convolve(rate1, kernel, mode='same')
        smoothed_rate2 = np.convolve(rate2, kernel, mode='same')

        return smoothed_rate1, smoothed_rate2

    rate /= smooth_win
    smoothed_rate = np.convolve(kernel, np.squeeze(rate), mode='same')

    return smoothed_rate, smoothed_rate


def interpolate_rates(rate_mon1, rate_mon2, time, time_interp, smooth_win):
    """downsample a population firing rate"""
    from scipy.interpolate import interp1d
    f_interp = [interp1d(time, rate.smooth_rate(window='flat', width=smooth_win)) for rate in [rate_mon1, rate_mon2]]
    return np.array([f(time_interp) for f in f_interp])


def choice_selection(task_info, monitors, downsample_step=10):
    # params
    sim_dt = unitless(task_info['sim']['sim_dt'], second, as_int=False)
    new_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False) * downsample_step
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    settle_time_idx = int(settle_time / new_dt)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    stim_off = unitless(task_info['sim']['stim_off'], second, as_int=False)
    smooth_win = task_info['sim']['smooth_win']
    rate_dec1, rate_dec2, rate_sen1, rate_sen2 = monitors

    # downsample population rates
    time = np.arange(0, runtime, sim_dt)
    time_low_def = np.arange(0, runtime, new_dt)
    rates_dec = interpolate_rates(rate_dec1, rate_dec2, time, time_low_def, smooth_win)
    rates_sen = interpolate_rates(rate_sen1, rate_sen2, time, time_low_def, smooth_win)

    # choice selection
    dec_ival = np.array([(stim_off-0.5)/new_dt, stim_off/new_dt], dtype=int)
    dec_winner = rates_dec[:, dec_ival[0]:dec_ival[1]].mean(axis=1)
    winner_pop = np.argmax(dec_winner)
    rates_dec = np.array([rates_dec[winner_pop, settle_time_idx:], rates_dec[~winner_pop, settle_time_idx:]])
    rates_sen = np.array([rates_sen[winner_pop, settle_time_idx:], rates_sen[~winner_pop, settle_time_idx:]])

    choice_data = {'rates_dec': rates_dec,
                   'rates_sen': rates_sen,
                   'winner_pop': np.array([winner_pop])}

    return choice_data


def reorder_winner_pop(pop_array):
    """case when second half of neurons belong to winner population"""
    sub = int(pop_array.shape[0] / 2)
    if sub > 1:
        pop_array = np.array([pop_array[sub:], pop_array[:sub]])
        return pop_array
    return np.array([pop_array[1], pop_array[0]])


def get_winner_loser_trials(rates, is_winner_pop):
    """divide all trials of a neuron into winner and loser"""
    winner_trials = rates[is_winner_pop]
    loser_trials = rates[np.logical_not(is_winner_pop)]

    return winner_trials, loser_trials


def choice_probability(winner_trials, loser_trials, step=10):
    """computes CP of a neuron from its spike distributions at each timepoint"""
    tps = winner_trials.shape[1]
    cp = np.empty(tps-step)     # int(tps/step)
    bins = np.arange(-1, 151)
    for t in range(tps-step):   # np.linspace(0, tps-step, int(tps/step), dtype=int)
        x1, e1 = np.histogram(winner_trials[:, t:t + step], bins=bins, density=True)
        x2, e2 = np.histogram(loser_trials[:, t:t + step], bins=bins, density=True)
        cp[t] = mtr.auc(np.cumsum(x1), np.cumsum(x2))

    return cp


def save_figure(task_dir, fig, fig_name, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(task_dir + fig_name)
    plt.close(fig)


def create_inset(axes, data2plt, c, xlim, w=1, h=0.7, nyticks=4):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(axes, w, h, loc=1)
    ax_ins.plot(data2plt[0], data2plt[1], color=c, lw=1.5)
    ax_ins.set_xlim(xlim)
    ax_ins.yaxis.get_major_locator().set_params(nbins=nyticks)
    plt.xticks(visible=False)

    return ax_ins


def plot_fig1(task_info, monitors, task_dir):
    sns.set(context='talk', style='darkgrid')

    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    smooth_win = task_info['sim']['smooth_win']
    spksSE, rateDE1, rateDE2, rateSE1, rateSE2, spksDE, rateDI, rateSI, stim1, stim2, stim_time = monitors
    subDE = int(spksDE.source.__len__() / 2)
    subSE = int(spksSE.source.__len__() / 2)
    nticks = 4

    fig1, axs = plt.subplots(7, 1, figsize=(8, 15), sharex=True)
    fig1.add_axes(axs[0])
    plt.title('Decision circuit')
    spksDE2 = spksDE.i >= subDE
    plot_raster(spksDE.i[spksDE2], spksDE.t[spksDE2], color='C0', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(subDE, 2*subDE, subDE/nticks))
    plt.ylabel(r'$Neuron$ $index$', {'horizontalalignment': 'right'})
    plt.xlim(settle_time, runtime)

    fig1.add_axes(axs[1])
    pos = axs[1].get_position()
    axs[1].set_position([pos.x0, pos.y0 + .01, pos.width, pos.height])
    spksDE1 = spksDE.i < subDE
    plot_raster(spksDE.i[spksDE1], spksDE.t[spksDE1], color='C3', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(0, subDE, subDE/nticks))
    plt.ylabel('')

    fig1.add_axes(axs[2])
    pos = axs[2].get_position()
    axs[2].set_position([pos.x0, pos.y0 + .02, pos.width, pos.height])
    plot_rate(rateDI.t, rateDI.smooth_rate(window='flat', width=smooth_win),
              color='C4', time_unit=second, lw=1, label='inh')
    plot_rate(rateDE1.t, rateDE1.smooth_rate(window='flat', width=smooth_win),
              color='C3', time_unit=second, lw=1.5, label='exc1')
    plot_rate(rateDE2.t, rateDE2.smooth_rate(window='flat', width=smooth_win),
              color='C0', time_unit=second, lw=1.5, label='exc2')
    plt.ylabel(r"$Rate$ (sp/s)")
    plt.ylim(0, 45)
    plt.yticks(np.arange(0, 55, 20))
    #plt.legend(loc='upper left', fontsize='x-small')

    fig1.add_axes(axs[3])
    plt.title('Sensory circuit')
    spksSE2 = spksSE.i >= subSE
    plot_raster(spksSE.i[spksSE2], spksSE.t[spksSE2], color='C0', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(subSE, 2*subSE, subSE/nticks))
    plt.ylabel(r'$Neuron$ $index$', {'horizontalalignment': 'right'})

    fig1.add_axes(axs[4])
    pos = axs[4].get_position()
    axs[4].set_position([pos.x0, pos.y0 + .01, pos.width, pos.height])
    spksSE1 = spksSE.i < subSE
    plot_raster(spksSE.i[spksSE1], spksSE.t[spksSE1], color='C3', marker='.', markersize=1, time_unit=second)
    plt.yticks(np.arange(0, subSE, subSE/nticks))
    plt.ylabel('')

    fig1.add_axes(axs[5])
    pos = axs[5].get_position()
    axs[5].set_position([pos.x0, pos.y0 + .02, pos.width, pos.height])
    plot_rate(rateSI.t, rateSI.smooth_rate(window='flat', width=smooth_win),
              color='C4', time_unit=second, lw=1, label='inh')
    plot_rate(rateSE1.t, rateSE1.smooth_rate(window='flat', width=smooth_win),
              color='C3', time_unit=second, lw=1.5, label='exc1')
    plot_rate(rateSE2.t, rateSE2.smooth_rate(window='flat', width=smooth_win),
              color='C0', time_unit=second, lw=1.5, label='exc2')
    plt.ylabel(r"$Rate$ (sp/s)")
    plt.ylim(0, 25)
    plt.yticks(np.arange(0, 30, 10))
    plt.legend(loc='upper left', fontsize='xx-small')

    fig1.add_axes(axs[6])
    plt.title('Stimulus')
    plt.plot(stim_time, stim1.mean(axis=0)*1e12, color='C3', lw=1.5)   # stim1.t, axis=0
    plt.plot(stim_time, stim2.mean(axis=0)*1e12, color='C0', lw=1.5)   # np.arange(0, 3.5, 1e-3), axis=1
    plt.xlabel(r"$Time$ (s)")
    plt.ylabel(r'$I_{soma}$ (pA)')
    for i in range(6):
        axs[i].set_xlabel('')

    save_figure(task_dir, fig1, '/figure1.png', tight=False)


def plot_fig2(task_info, events, bursts, spikes, stim1, stim2, stim_time, task_dir, fig_name='/figure2.png'):
    sns.set(context='talk', style='darkgrid')

    nn, tps = events.shape
    time = get_this_time(task_info, tps)
    new_dt = get_this_dt(task_info, tps)
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
    sub = int(nn / 2)

    fig2, axs = plt.subplots(5, 2, figsize=(16, 10), sharex=True, sharey='row')
    axs[0, 0].plot(stim_time, stim1.mean(axis=0)*1e12, color='C3', lw=1.5)
    axs[0, 0].set_title('Sensory population 1')
    axs[0, 0].set_ylabel(r'$I_{soma}$ (pA)')
    axs[0, 0].set_xlim(time[0], time[-1])
    axs[0, 1].plot(stim_time, stim2.mean(axis=0)*1e12, color='C0', lw=1.5)
    axs[0, 1].set_title('Sensory population 2')

    f_rate1, f_rate2 = smooth_rate(spikes, smooth_win, new_dt, sub)
    axs[1, 0].plot(time, f_rate1, lw=1.5, color='C5')
    axs[1, 0].set_ylabel(r'$A$ (Hz)')
    axs[1, 1].plot(time, f_rate2, lw=1.5, color='C5')

    b_rate1, b_rate2 = smooth_rate(bursts, smooth_win, new_dt, sub)
    axs[2, 0].plot(time, b_rate1, lw=1.5, color='C1')
    axs[2, 0].set_ylabel(r'$B$ (Hz)')
    axs[2, 1].plot(time, b_rate2, lw=1.5, color='C1')

    e_rate1, e_rate2 = smooth_rate(events, smooth_win, new_dt, sub)
    axs[4, 0].plot(time, e_rate1, lw=1.5, color='C2')
    axs[4, 0].set_ylabel(r'$E$ (Hz)')
    axs[4, 0].set_xlabel(r'$Time$ (s)')
    axs[4, 1].plot(time, e_rate2, lw=1.5, color='C2')
    axs[4, 1].set_xlabel(r'$Time$ (s)')

    bfracc1 = b_rate1 / e_rate1
    bfracc1[np.isnan(bfracc1)] = 0  # handle division by zero
    bfracc2 = b_rate2 / e_rate2
    bfracc2[np.isnan(bfracc2)] = 0  # handle division by zero
    axs[3, 0].plot(time, bfracc1*100, lw=1.5, color='C6')
    axs[3, 0].set_ylabel(r'$F$ (%)')
    axs[3, 1].plot(time, bfracc2*100, lw=1.5, color='C6')

    save_figure(task_dir, fig2, fig_name, tight=False)


def plot_fig3(task_info, dend_mon, events, bursts, spikes, task_dir):
    sns.set(context='talk', style='darkgrid')

    eta0 = unitless(task_info['plastic']['eta0'], pA)
    tauB = unitless(task_info['plastic']['tauB'], ms)
    tau_update = unitless(task_info['plastic']['tau_update'], ms)
    target = unitless(task_info['targetB'], Hz, as_int=False)
    step_update = int(task_info['plastic']['tau_update'] / task_info['sim']['sim_dt'])
    b_fb = task_info['bfb']
    B0 = unitless(task_info['plastic']['tauB'], second) * target
    eta = eta0 * tau_update / tauB
    fb_rate = unitless(task_info['plastic']['dec_winner_rate'], Hz)
    time = np.linspace(0, dend_mon.t_[-1], bursts.shape[1])
    last_time = time[-1]
    zoom_inteval = (last_time-10, last_time-5)
    xlim_inteval = (0, last_time)
    nn2plt = 10
    nrows, ncols = (4, 3)

    fig3, axs = plt.subplots(nrows, ncols, figsize=(int(8*ncols), int(5*nrows)), sharex='row')
    fig3.add_axes(axs[0, 0])
    plt.title(r'Plasticity weights')
    plt.plot(dend_mon.t_, dend_mon.muOUd[:nn2plt].T*1e12, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, dend_mon.muOUd.mean(axis=0)*1e12, color='C0', lw=1.5)
    plt.ylabel(r'$\mu_{OU_{d}}$ $(pA)$')
    plt.xlim(xlim_inteval)
    create_inset(axs[0, 0], (dend_mon.t_, dend_mon.muOUd.mean(axis=0) * 1e12), 'C0', zoom_inteval)

    fig3.add_axes(axs[1, 0])
    B = dend_mon.B.mean(axis=0)
    plt.title(r'Difference from target')
    plt.plot(dend_mon.t_[::step_update], dend_mon.B[:nn2plt, ::step_update].T - B0, color='gray', lw=0.5)
    plt.plot(dend_mon.t_, B - B0, color='C4', lw=1.5)
    plt.ylabel(r'$B - B0$')
    plt.xlim(xlim_inteval)
    create_inset(axs[1, 0], (dend_mon.t_, B - B0), 'C4', zoom_inteval)

    fig3.add_axes(axs[0, 1])
    plt.title(r'Dendritic background current')
    plt.plot(dend_mon.t_, dend_mon.Ibg[:nn2plt].T*1e9, color='gray', lw=0.2)
    plt.plot(dend_mon.t_, dend_mon.Ibg.mean(axis=0)*1e9, color='black', lw=1)
    plt.ylabel('$I_{OU_{d}}$ $(nA)$')
    plt.xlim(xlim_inteval)
    create_inset(axs[0, 1], (dend_mon.t_, dend_mon.Ibg.mean(axis=0)*1e9), 'black', zoom_inteval)

    fig3.add_axes(axs[1, 1])
    plt.title(r'Dendritic feedback current')
    plt.plot(dend_mon.t_, dend_mon.g_ea[:nn2plt].T*1e3, color='gray', lw=0.2)
    plt.plot(dend_mon.t_, dend_mon.g_ea.mean(axis=0)*1e3, color='C3', lw=1)
    plt.xlim(xlim_inteval)
    plt.ylabel(r'$g_{ea}$ $(a.u.)$ ${\sim}I_{dec}$')
    create_inset(axs[1, 1], (dend_mon.t_, dend_mon.g_ea.mean(axis=0)*1e3), 'C3', zoom_inteval)

    # plot neurometric params per subpopulation
    for i in range(events.shape[0]):
        fig3.add_axes(axs[2+i, 0])
        plt.plot(time, spikes[i], lw=1, color='C5')
        plt.ylabel(r'$A$ (Hz)')
        plt.xlim(xlim_inteval)
        create_inset(axs[2+i, 0], (time, spikes[i]), 'C5', zoom_inteval)

        fig3.add_axes(axs[2+i, 1])
        plt.plot(time, bursts[i], lw=1, color='C1')
        plt.axhline(target, color='gray', lw=2, ls='dashed')
        plt.ylabel(r'$B$ (Hz)')
        plt.xlim(xlim_inteval)
        create_inset(axs[2+i, 1], (time, bursts[i]), 'C1', zoom_inteval)

        fig3.add_axes(axs[2+i, 2])
        bfracc = bursts[i] / events[i]
        bfracc[np.isnan(bfracc)] = 0    # handle division by 0 - because no event also means no burst!
        plt.plot(time, bfracc*100, lw=1, color='C6')
        plt.ylabel(r'$F$ (%)')
        plt.xlim(xlim_inteval)
        plt.ylim(0, 100)
        create_inset(axs[2+i, 2], (time, bfracc*100), 'C6', zoom_inteval)

    for i in range(nrows):
        if i in [int((nrows-1)/2), nrows-1]:
            for j in range(ncols):
                axs[i, j].set_xlabel(r'$Time$ (s)')

    fig3.add_axes(axs[0, 2])
    plt.plot(0, c='white', label=r'target = %.1f Hz' % target)
    plt.plot(0, c='white', label=r'eta = %.4f pA' % eta)
    plt.plot(0, c='white', label=r'tauB = %i ms' % tauB)
    plt.plot(0, c='white', label=r'tau_update = %i ms' % tau_update)
    plt.plot(0, c='white', label=r'b_fb = %i' % b_fb)
    plt.plot(0, c='white', label=r'fb_rate = %i Hz' % fb_rate)
    plt.legend(loc='lower left', fontsize='large', frameon=False)
    plt.axis('off')
    plt.grid('off')
    fig3.add_axes(axs[1, 2])
    plt.axis('off')
    plt.grid('off')

    save_figure(task_dir, fig3, '/figure3.png')


def plot_plastic_rasters(task_info, spk_times, burst_times, bursts, task_dir):
    sns.set(context='talk', style='darkgrid')

    target = unitless(task_info['targetB'], Hz, as_int=False)
    last_time = unitless(task_info['sim']['runtime'], second)
    nn = len(spk_times)
    sub = min(int(nn / 2), 20)
    pre_time = 5  # in seconds
    interval = 5
    nticks = 4

    fig4, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=False)
    fig4.add_axes(axs[0])
    plt.title('Before plasticity')
    for n in range(sub):
        spks_pre = spk_times[n][spk_times[n] < pre_time]
        bursts_pre = burst_times[n][burst_times[n] < pre_time]
        plt.scatter(spks_pre, np.ones(len(spks_pre))*n, c='C0', marker='.', alpha=0.8, s=7)
        plt.scatter(bursts_pre, np.ones(len(bursts_pre))*n, c='C1', marker='.', s=7)
    plt.yticks(np.arange(0, sub+1, sub/nticks))
    plt.ylabel(r'$Neuron$ $index$')
    plt.xlim(pre_time-interval, pre_time)
    plt.xlabel(r'$Time$ $(s)$')

    fig4.add_axes(axs[1])
    plt.title('After plasticity')
    for n in range(sub):
        spks_post = spk_times[n][spk_times[n] > last_time-interval]
        bursts_post = burst_times[n][burst_times[n] > last_time-interval]
        plt.scatter(spks_post, np.ones(len(spks_post))*n, c='C0', marker='.', alpha=0.8, s=7)
        plt.scatter(bursts_post, np.ones(len(bursts_post))*n, c='C1', marker='.', s=7)
    plt.yticks(np.arange(0, sub+1, sub/nticks))
    plt.ylabel(r'$Neuron$ $index$')
    plt.xlim(last_time - interval, last_time)
    plt.xlabel(r'$Time$ $(s)$')

    fig4.add_axes(axs[2])
    maxbrate = 5.5
    plt.title('Plasticity rule accuracy')
    plt.plot(target, bursts[:int(bursts.shape[0]/2), int(last_time - 10):].mean(), c='C3', marker='o')
    plt.plot(np.linspace(0, maxbrate, 100), np.linspace(0, maxbrate, 100), c='gray', lw=1.5, ls='dashed')
    plt.xlabel(r'$Target$ $rate$ $(Hz)$')
    plt.ylabel(r'$Burst$ $rate$ $(Hz)$')
    plt.xlim(0, 5.5)
    plt.ylim(0, 5.5)

    save_figure(task_dir, fig4, '/figure4.png')


def plot_isis(task_info, isis, ieis, ibis, task_dir, bins=np.arange(0, 760, 10), extend_burst=2):
    sns.set(context='notebook', style='darkgrid')

    # params and different versions of cvs
    valid_burst = task_info['sim']['valid_burst']*1e3
    max_isi = int(bins[-1])
    step = bins[1]
    cv = isis.std() / isis.mean()
    cv_b = ibis.std() / ibis.mean()
    cv_e = ieis.std() / ieis.mean()

    fig5, axs = plt.subplots(1, 3, figsize=(12, 3), sharex=False, sharey=False)
    sns.distplot(isis[isis > valid_burst], kde=False, norm_hist=True, bins=bins, color='C5', ax=axs[0])
    axs[0].plot(0, c='white', label=r'CV = %.3f' % cv)
    axs[0].legend(loc='upper right')
    axs[0].set_title('Spikes')
    axs[0].set_ylabel(r'$Proportion$ $of$ $isi$')
    axs[0].set_xlim(valid_burst, max_isi)

    sns.distplot(ieis, kde=False, norm_hist=True, bins=bins, color='C2', ax=axs[1])
    axs[1].plot(0, c='white', label=r'CV_e = %.3f' % cv_e)
    axs[1].legend(loc='upper right')
    axs[1].set_title('Events')
    axs[1].set_xlabel(r'$Interspike$ $interval$ (ms)')
    axs[1].set_xlim(valid_burst, max_isi)

    burst_bins = np.arange(0, extend_burst * max_isi, extend_burst * step)
    sns.distplot(ibis, kde=False, norm_hist=True, bins=burst_bins, color='C1', ax=axs[2])
    axs[2].plot(0, c='white', label=r'CV_b = %.3f' % cv_b)
    axs[2].legend(loc='upper right')
    axs[2].set_title('Bursts')
    axs[2].set_xlim(extend_burst * valid_burst, extend_burst * max_isi)

    save_figure(task_dir, fig5, '/figure5.png')


def plot_pop_averages(task_info, rates_dec, rates_sen, cps, task_dir, fig_name='/fig1_averages.png'):
    sns.set(context='talk', style='darkgrid')

    # get params
    _, pops, tps1 = rates_dec.shape
    _, tps2 = cps.shape
    time1 = get_this_time(task_info, tps1)
    time2 = get_this_time(task_info, tps2)
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    stim_on = unitless(task_info['sim']['stim_on'], second, as_int=False) - settle_time
    stim_off = unitless(task_info['sim']['stim_off'], second, as_int=False) - settle_time

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    fig.add_axes(axs[0])
    plt.plot(time1, rates_dec[:, 0, :].mean(axis=0), c='C3', lw=2, label='pref')
    plt.plot(time1, rates_dec[:, 1, :].mean(axis=0), c='C0', lw=2, label='non-pref')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1.5)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1.5)
    plt.ylim(0, 50)
    plt.title('Integration circuit')
    plt.ylabel(r'$Population$ $rate$ (sp/sec)')

    fig.add_axes(axs[1])
    plt.title('Sensory circuit')
    plt.plot(time1, rates_sen[:, 0, :].mean(axis=0), c='C3', lw=2, label='pref')
    plt.plot(time1, rates_sen[:, 1, :].mean(axis=0), c='C0', lw=2, label='non-pref')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1.5)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1.5)
    plt.ylim(0, 25)
    plt.ylabel(r'$Population$ $rate$ (sp/sec)')
    plt.legend(loc='upper right', fontsize='x-small')

    fig.add_axes(axs[2])
    plt.plot(time2, cps.mean(axis=0), c='black', lw=2)
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1.5)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1.5)
    plt.xlabel(r'$Time$ (sec)')
    plt.ylabel(r'$Choice$ $prob.$')
    plt.ylim(0.41, 0.71)

    save_figure(task_dir, fig, fig_name)
