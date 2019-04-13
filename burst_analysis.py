import numpy as np
from brian2.units import second
from helper_funcs import unitless, handle_downsampled_spks, smooth_rate


def spk_times2neurometric_times(task_info, spk_mon):
    """Calculates burst, event and single times from SpikeMonitor.spike_times(), following Naud & Sprekeler 2018."""

    # params
    settle_time = unitless(task_info['sim']['settle_time'], second)
    valid_burst = task_info['sim']['valid_burst']
    mon_spk_times = spk_mon.spike_trains()
    nn = mon_spk_times.__len__()

    # allocate variables
    event_times = {}
    burst_times = {}
    single_times = {}
    spike_times = {}
    all_isis = np.zeros(1)

    for n in np.arange(nn, dtype=int):
        this_spks = unitless(mon_spk_times[n], second, as_int=False)
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


def neurometric_times2raster(task_info, all_spk_times, rate=False):
    """takes dictionaries of spk_times and transforms them to rasters or rates"""
    from scipy.sparse import lil_matrix

    # params
    event_times, burst_times, single_times, spk_times = all_spk_times
    new_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second)
    settle_time = unitless(task_info['sim']['settle_time'], second)
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False) / 2
    tps = unitless(int((runtime - settle_time)), new_dt)
    nn = spk_times.__len__()

    # allocate variables
    events = lil_matrix((nn, tps), dtype='float32')  # events
    bursts = lil_matrix((nn, tps), dtype='float32')  # bursts
    singles = lil_matrix((nn, tps), dtype='float32')  # single spikes
    spikes = lil_matrix((nn, tps), dtype='float32')  # normal, all spikes

    for n in np.arange(nn, dtype=int):
        # fill sparse matrix with the proper indices of the newdt
        events[n, handle_downsampled_spks(np.floor(event_times[n] / new_dt)).astype(int)] = 1
        bursts[n, handle_downsampled_spks(np.floor(burst_times[n] / new_dt)).astype(int)] = 1
        singles[n, handle_downsampled_spks(np.floor(single_times[n] / new_dt)).astype(int)] = 1
        spikes[n, handle_downsampled_spks(np.floor(spk_times[n] / new_dt)).astype(int)] = 1

    if rate:
        # from matrix2rate per subpopulation
        sub = int(nn / 2)
        rates = []
        for i, matrix in enumerate([events, bursts, singles, spikes]):
            rate1, rate2 = smooth_rate(matrix.toarray(), smooth_win, new_dt, sub)
            rates.append(np.vstack((rate1, rate2)))

        # event_rate, burst_rate, single_rate, firing_rate
        return rates[0], rates[1], rates[2], rates[3]

    return events.toarray(), bursts.toarray(), singles.toarray(), spikes.toarray()


# TODO: save spksperburst or not?
