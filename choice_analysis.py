from snep.utils import experiment_opener, filter_tasks
import numpy as np
import matplotlib.pyplot as plt
from brian2.units import *
import sklearn.metrics as mtr
from tqdm import tqdm
import pickle

plt_show = True
load_path = '~/Documents/WS19/MasterThesis/Experiments/run_hierarchical'


# helper functions
# def who_wins(task_dir):
#     originaltime = rateDE1.t / second
#     interptime = np.linspace(0, originaltime[-1], originaltime[-1] * 100)  # every 10 ms
#     fDE1 = interpolate.interp1d(originaltime, rateDE1.smooth_rate(window='flat', width=smooth_win))
#     fDE2 = interpolate.interp1d(originaltime, rateDE2.smooth_rate(window='flat', width=smooth_win))
#     fSE1 = interpolate.interp1d(originaltime, rateSE1.smooth_rate(window='flat', width=smooth_win))
#     fSE2 = interpolate.interp1d(originaltime, rateSE2.smooth_rate(window='flat', width=smooth_win))
#     rateDE = np.array([f(interptime) for f in [fDE1, fDE2]])
#     rateSE = np.array([f(interptime) for f in [fSE1, fSE2]])
#
#     # select the last half second of the stimulus
#     newdt = runtime_ / rateDE.shape[1]
#     settle_timeidx = int(settle_time_ / newdt)
#     dec_ival = np.array([(stim_off_ - 0.5) / newdt, stim_off_ / newdt], dtype=int)
#     who_wins = rateDE[:, dec_ival[0]:dec_ival[1]].mean(axis=1)
#
#     # divide trls into preferred and non-preferred
#     pref_msk = np.argmax(who_wins)
#     poprates_dec = np.array([rateDE[pref_msk], rateDE[~pref_msk]])  # 0: pref, 1: npref
#     poprates_sen = np.array([rateSE[pref_msk], rateSE[~pref_msk]])


def get_actn(rates, sub, nselect=100):
    """
    :param rates: 3d array containing the rates of a sensory neurons, shape (ntrls, neurons, time)
    :param sub:
    :param nselect:
    :return: 1d array with the indices of the selected active neurons, shape (100)
    """
    ntrls = rates.shape[0]
    nSE = rates.shape[1]
    m = rates.mean(axis=2)
    notsilent = np.nonzero(m)[1]
    acttrls = np.array([np.count_nonzero([notsilent == n]) for n in np.unique(notsilent)])
    actn = np.nonzero(acttrls == ntrls)[0]
    filteredactn1 = np.random.choice(actn[actn < int(nSE * sub)], size=nselect)
    filteredactn2 = np.random.choice(actn[actn >= int(nSE * sub)], size=nselect)
    filteredactn = np.vstack((filteredactn1, filteredactn2))

    return filteredactn


def get_CPs(rates, pref_msk, actn, dt, smoothwin=100e-3, step=5):
    """
    pref_msk is tailored to this population rates and it's a bool!
    get choice probability obtaining the distribution of each timepoint
    for active neurons of a population

    :param: rates: ntrls, nSE, timepoints
    :param: pref_msk:
    :param: actn:
    :param: dt:
    :param: smoothwin:
    :param: step:
    :return: aucs  # CPs
    """
    # params
    timepoints = rates.shape[2]
    this_time = np.linspace(0, timepoints, int(timepoints / step), dtype=int)[:-1]
    nselect = actn.shape[0]
    newdt = dt * step
    kernel = np.ones((int(smoothwin / newdt)))
    prefrates = rates[pref_msk == True, :, :]
    nprefrates = rates[pref_msk == False, :, :]

    # allocate variable to save CP
    aucs = np.zeros((nselect, this_time.shape[0]))
    smoothauc = aucs.copy()

    # for each neuron that is active
    for i, n in tqdm(enumerate(actn)):

        # define max rate of neuron
        maxrate = max(2, rates[:, n, :].max() + 1)

        # for each timepoint
        for j, t in enumerate(this_time):
            # get this rate across all trials
            pref = prefrates[:, n, t:t + step]
            npref = nprefrates[:, n, t:t + step]

            # hist
            x1, e1 = np.histogram(pref, bins=np.arange(maxrate), density=True)
            x2, e2 = np.histogram(npref, bins=np.arange(maxrate), density=True)

            # cumulative distribution
            cx1 = np.concatenate(([0], np.cumsum(x1)))
            cx2 = np.concatenate(([0], np.cumsum(x2)))

            # auc
            aucs[i, j] = mtr.auc(cx1, cx2)  # reversed because pref > npref

        smoothauc[i] = np.convolve(aucs[i], kernel, mode='same') / (smoothwin/newdt)

    return smoothauc


def get_corr(rates, actn, dt, smoothwin=250e-3, step=10):
    """
    Pearson correlations at each time point between two neurons
    bins are size step.

    :param: rates:
    :param: actn:
    :param: dt:
    :param: smoothwin:
    :param: step:
    :return: corrall, corrii, corrij
    """
    # params
    timepoints = rates.shape[2]
    this_time = np.linspace(0, timepoints, int(timepoints / step), dtype=int)[:-1]
    actn1 = actn[0]
    actn2 = actn[1]
    nselect = actn.shape[1]

    # allocate variables
    corrs = np.zeros((nselect ** 2 * 2, this_time.shape[0]))
    corrs1 = np.zeros((nselect ** 2, this_time.shape[0]))
    corrs2 = corrs1.copy()

    # massive for loop through both subpopulations
    for i1, n1 in tqdm(enumerate(actn1)):
        for i2, n2 in enumerate(actn2):
            for j, t in enumerate(this_time):

                # get rates for each case
                x11 = rates[:, n1, t:t + step].mean(axis=1)
                x12 = rates[:, actn1[i2], t:t + step].mean(axis=1)

                x21 = rates[:, actn2[i1], t:t + step].mean(axis=1)
                x22 = rates[:, n2, t:t + step].mean(axis=1)

                x1 = rates[:, n1, t:t + step].mean(axis=1)
                x2 = rates[:, n2, t:t + step].mean(axis=1)

                # check if we have info aside from zero
                if not (np.nonzero(x11)[0].size == False) or not (np.nonzero(x12)[0].size == False):
                    # correlations between pop1
                    corrs1[int(nselect * i1) + i2, j] = np.corrcoef(x11, x12)[0, 1]

                if not (np.nonzero(x21)[0].size == False) or not (np.nonzero(x22)[0].size == False):
                    # correlations between pop2
                    corrs2[int(nselect * i2) + i1, j] = np.corrcoef(x21, x22)[0, 1]

                if not (np.nonzero(x1)[0].size == False) or not (np.nonzero(x2)[0].size == False):
                    # correlations across pops)
                    k = np.corrcoef(x1, x2)
                    corrs[int(nselect * i1) + i2, j] = k[0, 1]
                    corrs[-int(nselect * i1) + i2, j] = k[1, 0]

    # return as corrsall, corrsii, corrsij
    corrsall = np.concatenate((corrs, corrs1, corrs2), axis=0)
    corrsii = np.concatenate((corrs1, corrs2), axis=0)

    # allocate variables for smoothing
    newdt = dt * step
    kernel = np.ones((int(smoothwin / newdt)))
    smoothcorrsall = np.zeros(corrsall.shape)
    smoothcorrsii = np.zeros(corrsii.shape)
    smoothcorrsij = np.zeros(corrs.shape)

    for n in np.arange(corrsall.shape[0]):
        smoothcorrsall[n] = np.convolve(corrsall[n], kernel, mode='same') / (smoothwin/newdt)

        if n < corrs.shape[0]:
            smoothcorrsii[n] = np.convolve(corrsii[n], kernel, mode='same') / (smoothwin/newdt)
            smoothcorrsij[n] = np.convolve(corrs[n], kernel, mode='same') / (smoothwin/newdt)

    return smoothcorrsall, smoothcorrsii, smoothcorrsij


# decorator for experiment
thisexperiment = '2018-12-11-09h04m25s'


@experiment_opener({'test0': thisexperiment}, load_path, show=plt_show)
def plot_fig2(tables_task_ids):
    """
    Using the experiment_opener decorator automates some of the tedious aspects of handling experiment
    files, including opening and closing the file, plus it also calls plt.show() if you ask it to.
    And finally, it fixes a problem with SVG files so that they don't explode Inkscape if you import them.

    :param tables_task_ids: dict mapping from user supplied name to a tuple of (tables, task_ids)
    :return:
    """
    from snep.tables.experiment import ExperimentTables

    tables, task_ids = tables_task_ids['test0']
    assert isinstance(tables, ExperimentTables)  # This allows PyCharm to autocomplete method names for tables
    params = tables.get_general_params(True)
    param_ranges = tables.read_param_ranges()

    # filter tasks to only the ones that reach the targets
    targets = [{('c',): 0, ('bfb',): 0}]
    target_ids = filter_tasks(task_ids, targets)

    # -------------------------------------
    # Get experiment results and params
    # -------------------------------------
    # Simulation times
    ntrls = len(target_ids)
    sub = params['sen']['populations']['sub']
    settletime = params['simulation']['settletime'] / second
    runtime = params['simulation']['runtime'] / second - settletime
    stimon = params['simulation']['stimon'] / second - settletime
    stimoff = params['simulation']['stimoff'] / second - settletime
    pops, timepoints = tables.get_raw_data(task_ids[0])['poprates_dec'].shape
    dt = runtime / timepoints
    nSE, downsampltimepoints = tables.get_computed(task_ids[0])['spikes'].shape
    time = np.linspace(0, runtime, timepoints)
    downsampltime = np.linspace(0, runtime, downsampltimepoints)
    downsampldt = runtime / downsampltimepoints

    # allocate variables
    rateDE = np.empty((ntrls, pops, timepoints), dtype='float32')
    rateSE = np.empty((ntrls, pops, timepoints), dtype='float32')
    spksSE = np.empty((ntrls, nSE, downsampltimepoints), dtype='float32')
    # evntSE = np.empty((ntrls, nSE, downsampltimepoints), dtype='float32')
    # brstSE = np.empty((ntrls, nSE, downsampltimepoints), dtype='float32')
    # snglSE = np.empty((ntrls, nSE, downsampltimepoints), dtype='float32')
    pref_msk = np.empty((ntrls, 1), dtype='int')

    # loop through trials and retrieve results
    for trl, tid in tqdm(enumerate(target_ids)):
        # get neurometric info of all neurons
        computed = tables.get_computed(tid)
        spksSE[trl] = computed['spikes']
        # evntSE[trl] = computed['events']
        # brstSE[trl] = computed['bursts']
        # snglSE[trl] = computed['singles']

        # population rates
        raw_data = tables.get_raw_data(tid)
        rateDE[trl] = raw_data['poprates_dec']  # 0: pref, 1: npref
        rateSE[trl] = raw_data['poprates_sen']  # 0: pref, 1: npref
        pref_msk[trl] = raw_data['pref_msk']

    # -------------------------------------
    # Choice probability and correlations
    # -------------------------------------
    # accuracy
    acc = pref_msk.sum() / ntrls

    # get active neurons, 100 per subpopulation
    actn = get_actn(spksSE, sub)

    # a calculation every 1, 5 or 10 ms?
    stepCP = 10
    auc1 = get_CPs(spksSE, np.logical_not(pref_msk), actn[0], downsampldt, step=stepCP)
    auc2 = get_CPs(spksSE, pref_msk.astype(bool), actn[1], downsampldt, step=stepCP)
    auc12 = np.concatenate((auc1, auc2), axis=0)

    stepCorr = 50
    corrsall, corrsii, corrsij = get_corr(spksSE, actn, downsampldt, step=stepCorr)

    # -------------------------------------
    # Plot figure 2
    # -------------------------------------
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    fig.add_axes(axs[0])
    plt.plot(time, rateDE[:, 0, :].mean(axis=0), c='C3', lw=2, label='preferred')
    plt.plot(time, rateDE[:, 1, :].mean(axis=0), c='C0', lw=2, label='non-preferred')
    plt.axvline(x=stimon, color='gray', ls='dashed', lw=1.5)
    plt.axvline(x=stimoff, color='gray', ls='dashed', lw=1.5)
    plt.title('Integration circuit')
    plt.ylabel('Population rate (sp/s)')  # , {'horizontalalignment': 'right'}
    plt.ylim(0, 50)
    # plt.legend(loc='center right', bbox_to_anchor=(1.22, 0.82))

    # sensory circuit
    fig.add_axes(axs[1])
    plt.plot(time, rateSE[:, 0, :].mean(axis=0), c='C3', lw=2, label='preferred')
    plt.plot(time, rateSE[:, 1, :].mean(axis=0), c='C0', lw=2, label='pon-preferred')
    plt.axvline(x=stimon, color='gray', ls='dashed', lw=2)
    plt.axvline(x=stimoff, color='gray', ls='dashed', lw=2)
    plt.title('Sensory circuit')
    plt.ylabel('Population rate (sp/s)')
    plt.ylim(0, 20)  # 0, 15
    plt.legend(loc='center', bbox_to_anchor=(0.76, 0.91), ncol=2, fontsize='x-small')

    # CPs
    # clean to plot
    aucm = auc12.mean(axis=0)
    ymin = 0.45
    cleanaucm = np.ones(aucm.shape) * np.nan
    cleanaucm[aucm > ymin] = aucm[aucm > ymin]

    fig.add_axes(axs[2])
    plt.plot(downsampltime[::stepCP][1:], cleanaucm, 'k', lw=2)
    plt.axvline(x=stimon, color='gray', ls='dashed', lw=2)
    plt.axvline(x=stimoff, color='gray', ls='dashed', lw=2)
    plt.ylabel('Choice prob.')
    plt.ylim(ymin, ymin + 0.2)  # ymin+0.2

    # correlations
    fig.add_axes(axs[3])
    plt.plot(downsampltime[::stepCorr][1:], np.nanmean(corrsall, axis=0), c='k', lw=2, label='EE')
    plt.plot(downsampltime[::stepCorr][1:], np.nanmean(corrsii, axis=0), c='C4', lw=2, label='EiEi')
    plt.plot(downsampltime[::stepCorr][1:], np.nanmean(corrsij, axis=0), c='C2', lw=2, label='EjEj')
    plt.axvline(x=stimon, color='gray', ls='dashed', lw=2)
    plt.axvline(x=stimoff, color='gray', ls='dashed', lw=2)
    plt.xlim(stimon - 0.5, stimoff + 0.5)
    plt.ylim(-0.2, 0.2)  # -0.25, 0.25
    plt.xlabel('Time (s)')
    plt.ylabel('Noise correlations')
    plt.legend(loc='center', bbox_to_anchor=(0.77, 0.95), ncol=3, fontsize='x-small')

    # save figure
    #savepath = '/Users/PSR/Documents/WS19/MasterThesis/Experiments/run_hierarchical/'
    fig.savefig(load_path + '/' + thisexperiment + '/figure2.png')
    plt.close(fig)

    # -------------------------------------
    # Save analysis
    # -------------------------------------
    thisanalysisname = '/CPs-' + str(ntrls) + 'trls-' + str(targets) + '.pkl'

    # save variables
    # with open(savepath + thisexperiment + thisanalysisname, 'wb') as f:
    #     pickle.dump([pref_msk,
    #                  actn,
    #                  auc12,
    #                  [corrsall, corrsii, corrsij]], f)

    # TODO: plot burst probability and coherence levels
    # TODO: plot accuracy!

# if __name__ == '__main__':
#     plot_fig2()
#     # explore_data()
