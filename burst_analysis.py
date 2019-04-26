import numpy as np
from brian2.units import second
from helper_funcs import unitless, handle_downsampled_spikes, smooth_rate


def spk_times2all_spk_times(task_info, mon_spk_times):
    """Calculates burst, event and single times from SpikeMonitor.spike_times(), following Naud & Sprekeler 2018."""

    # params
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    valid_burst = 16e-3
    nn = mon_spk_times.__len__()

    # allocate variables
    event_times = {}
    burst_times = {}
    single_times = {}
    spike_times = {}
    all_isis = np.zeros(1)

    for n in np.arange(nn, dtype=int):
        this_spks = mon_spk_times[n]
        this_spks = this_spks[this_spks >= settle_time]  # ignore spks during settle_time
        this_spks -= settle_time

        if len(this_spks) > 0:
            # count as burst if next spike is within 16 ms apart
            isis = np.diff(this_spks)
            all_isis = np.hstack((np.zeros(1), isis[isis > 0], all_isis))
            is_burst = np.concatenate(([False], isis < valid_burst)).astype(int)
            is_burst_bool = is_burst.astype(bool)
            is_event = np.logical_not(is_burst_bool).astype(int)
            nspks_per_burst = np.zeros(is_burst.shape)
            nburst = 0

            if is_burst.any():
                # add preceding burst
                start_burst = np.where(np.diff(is_burst) == True)[0]
                nburst = len(start_burst)
                is_burst[start_burst] = 1

                # count number of spikes in each burst
                ibi = np.concatenate((start_burst, is_burst.shape))  # inter-burst-intervals markers
                spks_per_burst = np.array([is_burst[ibi[b]:ibi[b + 1]].sum() for b in range(nburst)])
                spks_per_burst[spks_per_burst == 1] = 2  # sanity check, there's no 1 spk burst!
                is_burst[is_burst_bool] = 0  # rmv consecutive burst spks
                nspks_per_burst[is_burst.astype(bool)] = spks_per_burst

            # get single spks
            issingle = np.logical_and(is_burst == 0, is_event.astype(bool))

            # sanity check
            allspks = is_burst.sum() + issingle.sum() + nspks_per_burst.sum() - nburst
            assert allspks == len(this_spks), "ups, sth is weird in the burst quantification :("

            # get events, bursts, singles times
            event_times[n] = this_spks[is_event.astype(bool)]
            burst_times[n] = this_spks[is_burst.astype(bool)]
            single_times[n] = this_spks[issingle.astype(bool)]
            spike_times[n] = this_spks

    all_isis *= 1e3  # in ms
    return event_times, burst_times, single_times, spike_times, all_isis


def spk_mon2spk_times(task_info, spk_mon):
    """Calculates burst, event and single times from SpikeMonitor.spike_times(), following Naud & Sprekeler 2018."""

    # params
    settle_time = unitless(task_info['sim']['settle_time'], second, as_int=False)
    valid_burst = task_info['sim']['valid_burst']
    mon_spk_times = spk_mon.spike_trains()
    nn = mon_spk_times.__len__()

    # allocate variables
    event_times = {}
    burst_times = {}
    single_times = {}
    spike_times = {}
    all_isis = np.zeros(1)
    all_ibis = np.zeros(1)
    all_ieis = np.zeros(1)

    for n in np.arange(nn, dtype=int):
        this_spks = unitless(mon_spk_times[n], second, as_int=False)
        this_spks = this_spks[this_spks >= settle_time]  # ignore spks during settle_time
        this_spks -= settle_time

        if len(this_spks) > 0:
            # count as burst if next spike is within 16 ms apart
            isis = np.diff(this_spks)
            all_isis = np.hstack((np.zeros(1), isis[isis > 0]*1e3, all_isis))
            is_burst = np.concatenate(([False], isis < valid_burst)).astype(int)
            is_burst_bool = is_burst.astype(bool)
            is_event = np.logical_not(is_burst_bool).astype(int)
            nspks_per_burst = np.zeros(is_burst.shape)
            nburst = 0

            if is_burst.any():
                # add preceding burst
                start_burst = np.where(np.diff(is_burst) == True)[0]
                nburst = len(start_burst)
                is_burst[start_burst] = 1

                # count number of spikes in each burst
                ibi = np.concatenate((start_burst, is_burst.shape))  # inter-burst-intervals markers
                spks_per_burst = np.array([is_burst[ibi[b]:ibi[b + 1]].sum() for b in range(nburst)])
                spks_per_burst[spks_per_burst == 1] = 2  # sanity check, there's no 1 spk burst!
                is_burst[is_burst_bool] = 0  # rmv consecutive burst spks
                nspks_per_burst[is_burst.astype(bool)] = spks_per_burst

            # get single spks
            issingle = np.logical_and(is_burst == 0, is_event.astype(bool))

            # sanity check
            allspks = is_burst.sum() + issingle.sum() + nspks_per_burst.sum() - nburst
            assert allspks == len(this_spks), "Ups, sth is weird in the burst quantification :("

            # get events, bursts, singles times
            event_times[n] = this_spks[is_event.astype(bool)]
            burst_times[n] = this_spks[is_burst.astype(bool)]
            single_times[n] = this_spks[issingle.astype(bool)]
            spike_times[n] = this_spks

            # isis
            ieis = np.diff(event_times[n])
            ibis = np.diff(burst_times[n])
            all_ieis = np.hstack((np.zeros(1), ieis[ieis > 0]*1e3, all_ieis))
            all_ibis = np.hstack((np.zeros(1), ibis[ibis > 0]*1e3, all_ibis))
    all_isis = (all_isis[all_isis > 0], all_ieis[all_ieis > 0], all_ibis[all_ibis > 0])
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
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False)
    if broad_step:
        sim_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False)
    tps = unitless(int((runtime - settle_time)), sim_dt)
    nn = max(spk_times, key=int) + 1

    # allocate variables
    events = lil_matrix((nn, tps), dtype='float32')  # events
    bursts = lil_matrix((nn, tps), dtype='float32')  # bursts
    singles = lil_matrix((nn, tps), dtype='float32')  # single spikes
    spikes = lil_matrix((nn, tps), dtype='float32')  # normal, all spikes

    num_spikes = 0
    for n in spk_times.keys():
        # fill sparse matrix with the proper indices of the newdt
        events[n, handle_downsampled_spikes(np.floor(event_times[n] / sim_dt)).astype(int)] = 1
        bursts[n, handle_downsampled_spikes(np.floor(burst_times[n] / sim_dt)).astype(int)] = 1
        singles[n, handle_downsampled_spikes(np.floor(single_times[n] / sim_dt)).astype(int)] = 1
        spikes[n, handle_downsampled_spikes(np.floor(spk_times[n] / sim_dt)).astype(int)] = 1
        num_spikes += len(spk_times[n])

    if not broad_step:
        assert num_spikes == spikes.toarray().sum(), "Ups, you lost some spikes while creating the rasters."

    events = events.toarray()
    bursts = bursts.toarray()
    singles = singles.toarray()
    spikes = spikes.toarray()

    if rate:
        # from matrix2rate per subpopulation
        sub = int(nn / 2)
        rates = []
        for i, matrix in enumerate([events, bursts, singles, spikes]):
            rate1, rate2 = smooth_rate(matrix, smooth_win, sim_dt, sub)
            rates.append(np.vstack((rate1, rate2)))

        # event_rate, burst_rate, single_rate, firing_rate
        return rates[0], rates[1], rates[2], rates[3]

    elif downsample:
        # allocate variables
        count_window = 0.1      # 100 ms
        new_tps = int(tps / (count_window/sim_dt))
        events_downsample = np.empty((nn, new_tps), dtype='float32')
        bursts_downsample = np.empty((nn, new_tps), dtype='float32')
        singles_downsample = np.empty((nn, new_tps), dtype='float32')
        spikes_downsample = np.empty((nn, new_tps), dtype='float32')

        idxs = np.linspace(0, (runtime-settle_time)/sim_dt, int((runtime-settle_time)/count_window)+1, dtype=int)
        for i, idx1 in enumerate(idxs[:-1]):
            idx2 = idxs[i+1]
            events_downsample[:, i] = np.squeeze(events[:, idx1:idx2].sum(axis=1) / count_window)
            bursts_downsample[:, i] = np.squeeze(bursts[:, idx1:idx2].sum(axis=1) / count_window)
            singles_downsample[:, i] = np.squeeze(singles[:, idx1:idx2].sum(axis=1) / count_window)
            spikes_downsample[:, i] = np.squeeze(spikes[:, idx1:idx2].sum(axis=1) / count_window)

        return events, bursts, singles, spikes, \
               [events_downsample, bursts_downsample, singles_downsample, spikes_downsample]

    return events, bursts, singles, spikes


# TODO: save spksperburst or not?
