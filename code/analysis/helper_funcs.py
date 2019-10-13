import numpy as np
from scipy.signal import lfilter
from brian2.units import second, Hz, pA, ms, nS, mV
from brian2tools import plot_raster
from sklearn import metrics as mtr
import matplotlib.pyplot as plt
import seaborn as sns

cntxt = 'notebook'


def adjust_variable(var_prev, cm_prev, cm_new):
    """Adjusts conductances or current for different cell capacitances"""
    target = var_prev / cm_prev
    return target * cm_new


def unitless(quantity, time_unit, as_int=True):
    """Removes units from the quantity"""
    if as_int:
        return int(quantity / time_unit)
    return quantity / time_unit


def get_OUstim(n, tau, flip_stim=False):
    """Ornstein-Uhlenbeck process in discrete time"""
    a = np.exp(-(1 / tau))
    i = lfilter(np.ones(1), [1, -a], np.sqrt(1 - a**2) * np.random.randn(n))
    i = np.asanyarray(i, dtype=np.float32)
    if flip_stim:
        i = np.flip(i, axis=0)
    return i


def get_this_dt(task_params, tps, include_settle_time=False):
    runtime = unitless(task_params['sim']['runtime'], second, as_int=False)
    settle_time = unitless(task_params['sim']['settle_time'], second, as_int=False)
    if include_settle_time:
        return np.round(runtime/tps, decimals=4)
    return np.round((runtime - settle_time)/tps, decimals=4)


def get_this_time(task_params, tps, include_settle_time=False):
    settle_time = unitless(task_params['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_params['sim']['runtime'], second, as_int=False)
    if include_settle_time:
        return np.linspace(0, runtime, tps, dtype=np.float32)
    return np.linspace(0, runtime-settle_time, tps, dtype=np.float32)


def np_array(*args, **kw):
    if 'dtype' not in kw.keys():
        return np.array(*args, **kw, dtype=np.float32)
    else:
        return np.array(*args, **kw)


def handle_downsampled_spikes(spk_times):
    """transforms 2 spks in one dt to 2 consecutive spikes"""
    conflict = np.where(np.diff(spk_times) == 0)[0]
    spk_times[conflict] -= 1
    if np.any(np.diff(spk_times) == 0):     # recursive search for no 2 spikes in one dt bin
        spk_times = handle_downsampled_spikes(spk_times)

    return spk_times.astype(np.int16)


def instant_rate(task_info, spikes, smooth_win=None, step=10):
    """computes the instantaneous spike count for a neuron at each trl"""
    if not smooth_win:
        smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
    if len(spikes.shape) < 2:
        spikes = spikes[np.newaxis, :]
    n_trials, tps = spikes.shape
    new_dt = get_this_dt(task_info, tps)
    time = get_this_time(task_info, tps)
    time_low_def = get_this_time(task_info, int(tps/step))
    rates = np.empty((n_trials, int(tps/step)), dtype=np.float32)
    for i in range(n_trials):
        rate = smooth_rate(spikes[i], smooth_win, new_dt)
        rates[i] = interpolate_rates(rate, time, time_low_def)

    return rates


def smooth_rate(rate, smooth_win, dt, sub=[]):
    """rectangular sliding window on a firing rate to smooth"""
    kernel = np.ones(int(smooth_win / dt), dtype=np.float32)
    if bool(sub):
        rate1 = rate[:sub].mean(axis=0) / smooth_win
        rate2 = rate[sub:].mean(axis=0) / smooth_win
        smoothed_rate1 = np.convolve(rate1, kernel, mode='same').astype(np.float32)
        smoothed_rate2 = np.convolve(rate2, kernel, mode='same').astype(np.float32)
        return smoothed_rate1, smoothed_rate2

    rate /= smooth_win
    smoothed_rate = np.convolve(np.squeeze(rate), kernel, mode='same').astype(np.float32)
    return smoothed_rate


def interpolate_rates(rate, time, time_interp):
    """downsample a population firing rate"""
    from scipy.interpolate import interp1d
    f = interp1d(time, rate)
    return f(time_interp)


def choice_selection(task_info, monitors, downsample_step=10):
    # params
    sim_dt = unitless(task_info['sim']['sim_dt'], second, as_int=False)
    new_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False) * downsample_step
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    settle_time_idx = int(settle_time / new_dt)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    stim_off = unitless(task_info['sim']['stim_off'], second, as_int=False)
    smooth_win = task_info['sim']['smooth_win']

    # smooth rate monitors and downsample
    time = np.arange(0, runtime, sim_dt, dtype=np.float32)
    time_low_def = np.arange(0, runtime, new_dt, dtype=np.float32)
    smooth_rates = [rate.smooth_rate(window='flat', width=smooth_win) for rate in monitors]
    interp_rates = [interpolate_rates(rate, time, time_low_def) for rate in smooth_rates]
    rate_dec1, rate_dec2, rate_sen1, rate_sen2 = interp_rates
    rates_dec = np.vstack((rate_dec1, rate_dec2)).astype(np.float32)
    rates_sen = np.vstack((rate_sen1, rate_sen2)).astype(np.float32)

    # choice selection
    dec_ival = np_array([(stim_off-0.5)/new_dt, stim_off/new_dt], dtype=int)
    dec_winner = rates_dec[:, dec_ival[0]:dec_ival[1]].mean(axis=1)
    winner_pop = np.argmax(dec_winner)
    rates_dec = np_array([rates_dec[winner_pop, settle_time_idx:], rates_dec[~winner_pop, settle_time_idx:]])
    rates_sen = np_array([rates_sen[winner_pop, settle_time_idx:], rates_sen[~winner_pop, settle_time_idx:]])

    return rates_dec, rates_sen, np_array([winner_pop])


def get_winner_loser_trials(spikes, is_winner_pop):
    """divide all trials of a neuron into winner and loser"""
    winner_trials = spikes[is_winner_pop]
    loser_trials = spikes[np.logical_not(is_winner_pop)]

    return winner_trials, loser_trials


def reorder_winner_pop(pop_array, stack=False):
    """case when second half of neurons belong to winner population"""
    sub = int(pop_array.shape[0] / 2)
    if sub > 1:
        pop_array = np_array([pop_array[sub:], pop_array[:sub]])
        if stack:
            return np.vstack((pop_array[0], pop_array[1]))
        return pop_array
    return np_array([pop_array[1], pop_array[0]])


def choice_probability(winner_trials, loser_trials, step=1):
    """computes CP of a neuron from its spike distributions at each timepoint"""
    tps = winner_trials.shape[1]
    cp = np.empty(int(tps/step), dtype=np.float32)
    bins = np.arange(-1, 151)
    for t in np.linspace(0, tps-step, int(tps/step), dtype=np.int16):
        x1, e1 = np.histogram(winner_trials[:, t:t+step], bins=bins, density=True)
        x2, e2 = np.histogram(loser_trials[:, t:t+step], bins=bins, density=True)
        cp[int(t/step)] = mtr.auc(np.cumsum(x1), np.cumsum(x2))

    return cp


def pair_noise_corr(rates1, rates2, step=1):
    """computes Pearson correlation of rate1 and rate2 across trials at each timepoint"""
    tps = rates1.shape[-1]
    corr = np.empty(int(tps/step), dtype=np.float32)
    for t in np.linspace(0, tps-step, int(tps/step), dtype=np.int16):
        r1 = rates1[:, t:t+step].flatten()
        r2 = rates2[:, t:t+step].flatten()
        corr[int(t/step)] = np.corrcoef(r1, r2)[0, 1]
    return corr


def create_inset(axes, data2plt, c, xlim, w=1, h=0.7, nyticks=4):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(axes, w, h, loc=1)
    ax_ins.plot(data2plt[0], data2plt[1], color=c, lw=1.5)
    ax_ins.set_xlim(xlim)
    ax_ins.yaxis.get_major_locator().set_params(nbins=nyticks)
    plt.xticks(visible=False)

    return ax_ins


def save_figure(task_dir, fig, fig_name, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(task_dir + fig_name)
    plt.close(fig)


def plot_psychometric(stimuli, winner_pops, task_dir, fig_name):
    sns.set(context=cntxt, style='darkgrid')

    n_trials = winner_pops.shape[1]
    sem = winner_pops.std(axis=1)*100 / np.sqrt(n_trials)
    fig = plt.figure(figsize=(3, 2.5), dpi=300)
    plt.title('Psychometric curve')
    plt.errorbar(x=stimuli*100, y=winner_pops.mean(axis=1)*100, yerr=sem, fmt='.-', lw=1, capsize=1,
                 color='xkcd:plum', ecolor='xkcd:coral',)
    plt.xticks(np.linspace(min(stimuli), max(stimuli), 5)*100)
    plt.xlabel(r'$Coherence$ (%)')
    plt.ylabel(r'$\%$ $correct$')
    save_figure(task_dir, fig, fig_name)


def plot_fig1(task_info, monitors, task_dir, save_vars=False):
    sns.set(context=cntxt, style='darkgrid')
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    smooth_win = task_info['sim']['smooth_win']
    spksSE, rateDE1, rateDE2, rateSE1, rateSE2, spksDE, rateDI, rateSI, stim1, stim2, stim_time = monitors
    subDE = int(spksDE.source.__len__() / 2)
    subSE = int(spksSE.source.__len__() / 2)
    nticks = 4
    nrows, ncols = (7, 1)

    fig1, axs = plt.subplots(nrows, ncols, figsize=(int(8 * ncols), int(2 * nrows)), dpi=100, sharex=True)
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
    rate_t = np_array(rateDI.t[:])
    rate_DI = np_array(rateDI.smooth_rate(window='flat', width=smooth_win))
    rate_DE1 = np_array(rateDE1.smooth_rate(window='flat', width=smooth_win))
    rate_DE2 = np_array(rateDE2.smooth_rate(window='flat', width=smooth_win))
    axs[2].plot(rate_t, rate_DI, color='C4', lw=1, label='inh')
    axs[2].plot(rate_t, rate_DE1, color='C3', lw=1.5, label='E1')
    axs[2].plot(rate_t, rate_DE2, color='C0', lw=1.5, label='E2')
    plt.ylabel(r"$Rate$ (sp/s)")
    plt.ylim(0, 45)
    plt.yticks(np.arange(0, 50, 15))
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
    rate_SI = np_array(rateSI.smooth_rate(window='flat', width=smooth_win))
    rate_SE1 = np_array(rateSE1.smooth_rate(window='flat', width=smooth_win))
    rate_SE2 = np_array(rateSE2.smooth_rate(window='flat', width=smooth_win))
    axs[5].plot(rate_t, rate_SI, color='C4', lw=1, label='inh')
    axs[5].plot(rate_t, rate_SE1, color='C3', lw=1.5, label='E1')
    axs[5].plot(rate_t, rate_SE2, color='C0', lw=1.5, label='E2')
    plt.ylabel(r"$Rate$ (sp/s)")
    plt.ylim(0, 20)
    plt.yticks(np.arange(0, 20, 5))
    plt.legend(loc='upper left', fontsize='xx-small')

    fig1.add_axes(axs[6])
    plt.title('Stimulus')
    stim1 = np_array(stim1.mean(axis=0)) * 1e12
    stim2 = np_array(stim2.mean(axis=0)) * 1e12
    plt.plot(stim_time, stim1, color='C3', lw=1.5)   # stim1.t, axis=0
    plt.plot(stim_time, stim2, color='C0', lw=1.5)   # np.arange(0, 3.5, 1e-3), axis=1
    plt.xlabel(r"$Time$ (s)")
    plt.ylabel(r'$I_{soma}$ (pA)')
    for i in range(6):
        axs[i].set_xlabel('')

    save_figure(task_dir, fig1, '/figure1.png', tight=False)

    # save variables in dict
    if save_vars:
        sim_state = {'spksi_dec': np_array(spksDE.i[:]), 'spkst_dec': np_array(spksDE.t[:]),
                     'spksi_sen': np_array(spksSE.i[:]), 'spkst_sen': np_array(spksSE.t[:]),
                     'rate_t': rate_t, 'rate_DE1': rate_DE1, 'rate_DE2': rate_DE2, 'rate_DI': rate_DI,
                     'rate_SE1': rate_SE1, 'rate_SE2': rate_SE2, 'rate_SI': rate_SI,
                     'stim1': stim1, 'stim2': stim2, 'stim_time': stim_time}
        return sim_state


def plot_fig2(task_info, events, bursts, spikes, stim_diff, stim_time, rates_dec, winner_pop, task_dir, fig_name='/figure2.png'):
    sns.set(context=cntxt, style='darkgrid')

    nn, tps = events.shape
    sub = int(nn / 2)
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
    time = get_this_time(task_info, tps)
    new_dt = get_this_dt(task_info, tps)
    stim_tps = stim_diff.shape[0]
    if tps != stim_tps:
        stim_dt = get_this_dt(task_info, stim_tps, include_settle_time=True)
        settle_time_idx = int(unitless(task_info['sim']['settle_time'], second, as_int=False) / stim_dt)
        stim_diff = stim_diff[settle_time_idx:]
        stim_time = stim_time[:stim_diff.shape[-1]]
    if winner_pop:
        events = reorder_winner_pop(events, stack=True)
        bursts = reorder_winner_pop(bursts, stack=True)
        spikes = reorder_winner_pop(spikes, stack=True)
    tps2 = rates_dec.shape[-1]
    time2 = get_this_time(task_info, tps2)
    n_dec = task_info['dec']['N_E'] * task_info['dec']['sub']
    b_fb = task_info['bfb']
    gleak = 24.2857*nS
    g = (0.004*b_fb*gleak)
    v_dend = 73*mV
    nrows, ncols = (3, 2)

    fig2, axs = plt.subplots(nrows, ncols, figsize=(int(6 * ncols), int(2 * nrows)), dpi=100, sharex=True)
    stim_kernel = smooth_rate(stim_diff, smooth_win, new_dt)
    axs[0, 0].plot(stim_time, stim_kernel, color='black', lw=1.5)
    axs[0, 0].set_ylabel(r'$stim$ $strength$ (au)')
    axs[0, 0].set_xlim(time[0], time[-1])

    f_rate1, f_rate2 = smooth_rate(spikes, smooth_win, new_dt, sub)
    axs[1, 0].plot(time, f_rate1, lw=1.5, color='C2', label='pref')
    axs[1, 0].plot(time, f_rate2, lw=1.5, color='gray', label='non-pref')
    axs[1, 0].set_ylabel(r'$A$ (Hz)')
    axs[1, 0].set_ylim(0, 15)

    e_rate1, e_rate2 = smooth_rate(events, smooth_win, new_dt, sub)
    axs[2, 0].plot(time, e_rate1, lw=1.5, color='xkcd:cerulean', label='pref')
    axs[2, 0].plot(time, e_rate2, lw=1.5, color='gray', label='non-pref')
    axs[2, 0].set_ylabel(r'$E$ (Hz)')
    axs[2, 0].set_xlabel(r'$Time$ (s)')
    axs[2, 0].set_ylim(0, 15)

    top_down1 = n_dec * 0.2 * rates_dec[0] * (1*ms) * g * v_dend
    top_down2 = n_dec * 0.2 * rates_dec[1] * (1*ms) * g * v_dend
    axs[0, 1].plot(time2, top_down1/pA, color='C3', lw=1.5, label='pref')
    axs[0, 1].plot(time2, top_down2/pA, color='C0', lw=1.5, label='non-pref')
    axs[0, 1].set_ylabel(r'$I_{top-down}$ (pA)')

    b_rate1, b_rate2 = smooth_rate(bursts, smooth_win, new_dt, sub)
    axs[1, 1].plot(time, b_rate1, lw=1.5, color='C1', label='pref')
    axs[1, 1].plot(time, b_rate2, lw=1.5, color='gray', label='non-pref')
    axs[1, 1].set_ylabel(r'$B$ (Hz)')
    axs[1, 1].set_ylim(0, 4)

    bf1 = b_rate1 / e_rate1
    bf1[np.isnan(bf1)] = 0  # handle division by zero
    bf2 = b_rate2 / e_rate2
    bf2[np.isnan(bf2)] = 0  # handle division by zero
    axs[2, 1].plot(time, bf1*100, lw=1.5, color='xkcd:light red', label='pref')
    axs[2, 1].plot(time, bf2*100, lw=1.5, color='gray', label='non-pref')
    axs[2, 1].set_ylabel(r'$F$ (%)')
    axs[2, 1].set_xlabel(r'$Time$ (s)')
    axs[2, 1].set_ylim(0, 50)

    for r in range(nrows):
        axs[r, 1].legend(loc='best', ncol=2, fontsize='xx-small')
    save_figure(task_dir, fig2, fig_name, tight=True)


def plot_isis(task_info, isis, ieis, ibis, cvs, spks_per_burst, task_dir, fig_name='/figure5.png',
              bins=np.arange(0, 760, 10), extend_burst=2):
    sns.set(context=cntxt, style='darkgrid')
    valid_burst = task_info['sim']['valid_burst']*1e3
    max_isi = int(bins[-1])
    step = bins[1]
    cv = cvs.mean()
    sp_burst = spks_per_burst.mean()
    nrows, ncols = (2, 3)

    fig5, axs = plt.subplots(nrows, ncols, figsize=(int(4*ncols), int(3*nrows)), dpi=100, sharex=False, sharey='row')
    sns.distplot(isis[isis > valid_burst], bins=bins, kde=False, norm_hist=True, color='C5', ax=axs[0, 0])
    axs[0, 0].set_title('spikes')
    axs[0, 0].set_ylabel(r'$Proportion$')
    axs[0, 0].set_xlim(valid_burst, max_isi)

    sns.distplot(ieis, bins=bins, kde=False, norm_hist=True, color='C2', ax=axs[0, 1])
    axs[0, 1].set_title('events')
    axs[0, 1].set_xlabel(r'$Interspike$ $interval$ (ms)')
    axs[0, 1].set_xlim(valid_burst, max_isi)

    burst_bins = np.arange(0, extend_burst * max_isi, extend_burst * step)
    sns.distplot(ibis, bins=burst_bins, kde=False, norm_hist=True, color='C1', ax=axs[0, 2])
    axs[0, 2].set_title('bursts')
    axs[0, 2].set_xlim(extend_burst * valid_burst, extend_burst * max_isi)

    sns.distplot(cvs, bins=np.linspace(0, 2, 50), kde=False, norm_hist=True, color='C4', ax=axs[1, 0])
    axs[1, 0].set_title(r'CV = %.3f' % cv)
    axs[1, 0].set_ylabel(r'$Proportion$')
    axs[1, 0].set_xlabel(r'$CV$ $per$ $neuron$')

    sns.distplot(spks_per_burst, bins=range(2, 7, 1), kde=False, norm_hist=True, color='C3', ax=axs[1, 1])
    axs[1, 1].set_title(r'sp/burst = %.3f' % sp_burst)
    axs[1, 1].set_xlabel(r'$No.$ $spikes$ $per$ $burst$')

    axs[1, 2].set_axis_off()

    save_figure(task_dir, fig5, fig_name)


def plot_fig3(task_info, dend_mon, events, bursts, spikes, pop_dend, task_dir):
    sns.set(context=cntxt, style='darkgrid')
    smooth_win = task_info['sim']['smooth_win']
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
    zoom_inteval = (last_time-5, last_time-3)
    xlim_inteval = (0, last_time)
    nn2plt = 10
    nrows, ncols = (4, 3)

    fig3, axs = plt.subplots(nrows, ncols, figsize=(int(6*ncols), int(4*nrows)), dpi=100, sharex='row')
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
    plt.plot(dend_mon.t_[::step_update], dend_mon.Ibg[:nn2plt, ::step_update].T*1e9, color='gray', lw=0.2)
    plt.plot(dend_mon.t_[::step_update], dend_mon.Ibg[:, ::step_update].mean(axis=0)*1e9, color='black', lw=1)
    plt.ylabel('$I_{OU_{d}}$ $(nA)$')
    plt.xlim(xlim_inteval)
    create_inset(axs[0, 1], (dend_mon.t_[::step_update], dend_mon.Ibg[:, ::step_update].mean(axis=0)*1e9), 'black', zoom_inteval)

    fig3.add_axes(axs[1, 1])
    plt.title(r'Dendritic feedback current')
    plt.plot(dend_mon.t_[::step_update], dend_mon.g_ea[:nn2plt, ::step_update].T*1e3, color='gray', lw=0.2)
    plt.plot(dend_mon.t_[::step_update], dend_mon.g_ea[:, ::step_update].mean(axis=0)*1e3, color='C3', lw=1)
    plt.xlim(xlim_inteval)
    plt.ylabel(r'$g_{ea}$ $(a.u.)$ ${\sim}I_{dec}$')
    create_inset(axs[1, 1], (dend_mon.t_[::step_update], dend_mon.g_ea[:, ::step_update].mean(axis=0)*1e3), 'C3', zoom_inteval)

    fig3.add_axes(axs[2, 1])
    plt.plot(pop_dend.t, pop_dend.smooth_rate(window='flat', width=smooth_win), color='C4', lw=1)

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

    fig3.add_axes(axs[1, 2])
    for label, lab_var in zip([r'target = %.1f Hz', r'eta = %.4f pA', r'tauB = %i ms',
                               'tau_update = %i ms', r'b_fb = %i', r'fb_rate = %i Hz'],
                              [target, eta, tauB, tau_update, b_fb, fb_rate]):
        plt.plot(0, c='white', label=label % lab_var)
    plt.legend(loc='center left', fontsize='small', frameon=False)
    plt.axis('off')
    axs[0, 2].set_axis_off()

    save_figure(task_dir, fig3, '/figure3.png', tight=True)


def plot_plastic_rasters(task_info, spk_times, burst_times, bursts, task_dir):
    sns.set(context=cntxt, style='darkgrid')
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


def plot_plastic_check(task_info, pop_dend1, spks_dend, bursts, burst_times, task_dir):
    sns.set(context=cntxt, style='darkgrid')
    smooth_win = task_info['sim']['smooth_win']
    target = unitless(task_info['targetB'], Hz, as_int=False)
    last_time = unitless(task_info['sim']['runtime'], second)
    time = np.linspace(0, last_time, bursts.shape[1])
    nn = len(burst_times)
    sub = max(int(nn / 2), 20)  # min for only 20
    interval = 5

    fig5, axs = plt.subplots(2, 1, figsize=(8, 6))
    fig5.add_axes(axs[0])
    plt.plot(pop_dend1.t, pop_dend1.smooth_rate(window='flat', width=smooth_win), color='C4', lw=1, label='dend mon')
    plt.plot(time, bursts[0], lw=1, color='C1', label='burst quant')
    plt.axhline(target, color='gray', lw=2, ls='dashed')
    plt.legend(loc='best', fontsize='xx-small', ncol=2)
    plt.ylabel(r'rate (Hz)')

    fig5.add_axes(axs[1])
    for n in range(sub):
        bursts = burst_times[n][burst_times[n] > last_time - interval]
        plt.scatter(bursts, np.ones(len(bursts)) * n, c='C1', marker='.', s=10, alpha=0.75)
    spks_sub = spks_dend.i < sub
    plot_raster(spks_dend.i[spks_sub], spks_dend.t[spks_sub], color='C4', marker='.', markersize=5, time_unit=second, alpha=0.5)
    plt.xlim(last_time - interval, last_time)
    plt.xlabel('time (s)')
    plt.ylabel('neuron index')

    save_figure(task_dir, fig5, '/fig5-sanity_check.png', tight=True)


def plot_pop_averages(task_info, rates_dec, rates_sen, all_cps, corr_ii, corr_ij, task_dir, fig_name='/fig1_averages.png'):
    sns.set(context=cntxt, style='darkgrid')
    cps, e_cps, bf_cps = all_cps
    _, pops, tps1 = rates_dec.shape
    tps2 = cps.shape[-1]
    tps3 = corr_ii.shape[-1]
    time1 = get_this_time(task_info, tps1)
    time2 = get_this_time(task_info, tps2)
    time3 = get_this_time(task_info, tps3)
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    stim_on = unitless(task_info['sim']['stim_on'], second, as_int=False) - settle_time
    stim_off = unitless(task_info['sim']['stim_off'], second, as_int=False) - settle_time
    nrows, ncols = (4, 1)

    fig, axs = plt.subplots(nrows, ncols, figsize=(int(8*ncols), int(2*nrows)), dpi=100, sharex=True)
    fig.add_axes(axs[0])
    plt.plot(time1, rates_dec[:, 0, :].mean(axis=0), c='C3', lw=1.5, label='pref')
    plt.plot(time1, rates_dec[:, 1, :].mean(axis=0), c='C0', lw=1.5, label='non-pref')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1)
    plt.ylim(0, 40)
    plt.title('Integration circuit')
    plt.ylabel(r'$Population$ $rate$ (sp/sec)', {'horizontalalignment': 'right'})

    fig.add_axes(axs[1])
    plt.title('Sensory circuit')
    plt.plot(time1, rates_sen[:, 0, :].mean(axis=0), c='C3', lw=1.5, label='pref')
    plt.plot(time1, rates_sen[:, 1, :].mean(axis=0), c='C0', lw=1.5, label='non-pref')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1)
    plt.ylim(0, 15)
    plt.legend(loc='upper right', ncol=2, fontsize='x-small')

    fig.add_axes(axs[2])
    plt.plot(time2, cps.mean(axis=0), c='black', lw=1.5, label='A')
    plt.plot(time2, e_cps.mean(axis=0), c='xkcd:cerulean', lw=1.5, label='E')
    plt.plot(time2, bf_cps.mean(axis=0), c='xkcd:light red', lw=1.5, label='F')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1)
    plt.legend(loc='upper right', ncol=3, fontsize='x-small')
    plt.ylabel(r'$Choice$ $prob.$')
    plt.ylim(0.41, 0.61)

    fig.add_axes(axs[3])
    corr_all = np.vstack((corr_ii, corr_ij))
    plt.plot(time3, np.nanmean(corr_all, axis=0), c='black', lw=1.5, label='EE')
    plt.plot(time3, np.nanmean(corr_ii, axis=0), c='xkcd:magenta', lw=1.5, label='EiEi')
    plt.plot(time3, np.nanmean(corr_ij, axis=0), c='xkcd:turquoise', lw=1.5, label='EiEj')
    plt.axvline(x=stim_on, color='gray', ls='dashed', lw=1)
    plt.axvline(x=stim_off, color='gray', ls='dashed', lw=1)
    plt.legend(loc='upper right', ncol=3, fontsize='x-small')
    plt.xlabel(r'$Time$ (sec)')
    plt.ylabel(r'$Correlation$')

    save_figure(task_dir, fig, fig_name, tight=False)
