import numpy as np
from brian2.units import second
from helper_funcs import unitless, handle_downsampled_spikes, smooth_rate


def spk_mon2spk_times(task_info, spk_mon, nn2rec=50):
    """Calculates burst, event and single times from SpikeMonitor.spike_times(), following Naud & Sprekeler 2018."""

    # params
    settle_time = task_info['sim']['settle_time']
    valid_burst = task_info['sim']['valid_burst']*1e3
    mon_spk_times = spk_mon.spike_trains()
    sub = int(task_info['sen']['N_E'] * task_info['sen']['sub'])

    # random selection of active neurons
    active_n = np.array([n for n, spks in mon_spk_times.items() if len(spks[spks >= settle_time]) >= 3])
    nn_rec1 = np.random.choice(active_n[active_n < sub], size=nn2rec)
    nn_rec2 = np.random.choice(active_n[active_n >= sub], size=nn2rec)
    nn_rec = np.hstack((nn_rec1, nn_rec2))

    # allocate variables
    event_times = {}
    burst_times = {}
    single_times = {}
    spike_times = {}
    all_isis = np.zeros(1, dtype=np.float32)
    all_ibis = np.zeros(1, dtype=np.float32)
    all_ieis = np.zeros(1, dtype=np.float32)
    cvs = []
    mean_spks_per_burst = []

    for i, n in enumerate(nn_rec):
        # ignore spks during settle_time and work with unitless spike_times
        this_spks = mon_spk_times[n].astype(np.float32)
        this_spks = this_spks[this_spks >= settle_time] - settle_time
        this_spks /= second

        isis = np.diff(this_spks)
        isis = isis[isis > 0]*1e3
        all_isis = np.hstack((np.zeros(1), isis, all_isis)).astype(np.float32)
        is_burst = np.concatenate(([False], isis < valid_burst)).astype(np.int16)
        is_burst_bool = is_burst.astype(bool)
        is_event = np.logical_not(is_burst_bool).astype(np.int16)
        nburst = 0
        spks_per_burst = np.zeros(1, dtype='float32')

        if is_burst.any():
            cv = isis.std() / isis.mean()
            if not np.isnan(cv) and cv > 0:
                cvs.append(cv)

            # add preceding burst
            start_burst = np.where(np.diff(is_burst) == True)[0]
            nburst = len(start_burst)
            is_burst[start_burst] = 1

            # count number of spikes in each burst
            ibi = np.concatenate((start_burst, is_burst.shape))  # inter-burst-intervals markers
            spks_per_burst = np.array([is_burst[ibi[b]:ibi[b + 1]].sum() for b in range(nburst)], dtype='float32')
            spks_per_burst[spks_per_burst == 1] = 2  # sanity check, there's no 1 spk burst!
            is_burst[is_burst_bool] = 0  # rmv consecutive burst spks
            _ = [mean_spks_per_burst.append(spb) for spb in spks_per_burst]

        # get single spks
        issingle = np.logical_and(is_burst == 0, is_event.astype(bool))

        # sanity check
        allspks = is_burst.sum() + issingle.sum() + spks_per_burst.sum() - nburst
        assert allspks == len(this_spks), "Ups, sth is weird in the burst quantification :("

        # get events, bursts, singles times
        event_times[i] = this_spks[is_event.astype(bool)]
        burst_times[i] = this_spks[is_burst.astype(bool)]
        single_times[i] = this_spks[issingle.astype(bool)]
        spike_times[i] = this_spks

        # isis in ms
        ieis = np.diff(event_times[i])
        ibis = np.diff(burst_times[i])
        all_ieis = np.hstack((np.zeros(1), ieis[ieis > 0]*1e3, all_ieis)).astype(np.float32)
        all_ibis = np.hstack((np.zeros(1), ibis[ibis > 0]*1e3, all_ibis)).astype(np.float32)
    all_isis = (all_isis[all_isis > 0], all_ieis[all_ieis > 0], all_ibis[all_ibis > 0],
                np.array(cvs, dtype=np.float32), np.array(mean_spks_per_burst, dtype=np.float32))
    all_spk_times = (event_times, burst_times, single_times, spike_times)

    return all_spk_times, all_isis


def spk_times2raster(task_info, all_spk_times, broad_step=False, rate=False, downsample=False):
    """takes dictionaries of spk_times and transforms them to rasters or rates"""
    from scipy.sparse import lil_matrix

    # params
    event_times, burst_times, single_times, spk_times = all_spk_times
    sim_dt = unitless(task_info['sim']['sim_dt'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second, as_int=False)
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    if broad_step:
        sim_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False)
    tps = unitless(int((runtime - settle_time)), sim_dt)
    nn = max(spk_times, key=int) + 1

    # allocate variables
    events = lil_matrix((nn, tps), dtype=np.float32)  # events
    bursts = lil_matrix((nn, tps), dtype=np.float32)  # bursts
    singles = lil_matrix((nn, tps), dtype=np.float32)  # single spikes
    spikes = lil_matrix((nn, tps), dtype=np.float32)  # normal, all spikes

    num_spikes = 0
    for n in spk_times.keys():
        # fill sparse matrix with the proper indices of the newdt
        events[n, handle_downsampled_spikes(np.floor(event_times[n] / sim_dt))] = 1
        bursts[n, handle_downsampled_spikes(np.floor(burst_times[n] / sim_dt))] = 1
        singles[n, handle_downsampled_spikes(np.floor(single_times[n] / sim_dt))] = 1
        spikes[n, handle_downsampled_spikes(np.floor(spk_times[n] / sim_dt))] = 1
        num_spikes += len(spk_times[n])

    if not broad_step:
        assert num_spikes == spikes.toarray().sum(), "Ups, you lost some spikes while creating the rasters."

    if rate:
        # from matrix2rate per subpopulation
        smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
        sub = int(nn / 2)
        rates = []
        for i, matrix in enumerate([events, bursts, singles, spikes]):
            rate1, rate2 = smooth_rate(matrix.toarray(), smooth_win, sim_dt, sub)
            rates.append(np.vstack((rate1, rate2)))

        # event_rate, burst_rate, single_rate, firing_rate
        return rates[0], rates[1], rates[2], rates[3]

    elif downsample:
        # allocate variables
        count_window = 0.1      # 100 ms
        new_tps = int(tps / (count_window/sim_dt))
        events_downsample = np.empty((nn, new_tps), dtype=np.float32)
        bursts_downsample = np.empty((nn, new_tps), dtype=np.float32)
        singles_downsample = np.empty((nn, new_tps), dtype=np.float32)
        spikes_downsample = np.empty((nn, new_tps), dtype=np.float32)

        idxs = np.linspace(0, (runtime-settle_time)/sim_dt, int((runtime-settle_time)/count_window)+1, dtype=np.int16)
        for i, idx1 in enumerate(idxs[:-1]):
            idx2 = idxs[i+1]
            events_downsample[:, i] = np.squeeze(events.toarray()[:, idx1:idx2].sum(axis=1) / count_window)
            bursts_downsample[:, i] = np.squeeze(bursts.toarray()[:, idx1:idx2].sum(axis=1) / count_window)
            singles_downsample[:, i] = np.squeeze(singles.toarray()[:, idx1:idx2].sum(axis=1) / count_window)
            spikes_downsample[:, i] = np.squeeze(spikes.toarray()[:, idx1:idx2].sum(axis=1) / count_window)

        return events.toarray(), bursts.toarray(), singles.toarray(), spikes.toarray(), \
               [events_downsample, bursts_downsample, singles_downsample, spikes_downsample]

    return events.toarray(), bursts.toarray(), singles.toarray(), spikes.toarray()
