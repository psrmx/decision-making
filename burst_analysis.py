import numpy as np
from brian2.units import second
from helper_funcs import unitless, handle_downsampled_spks, smooth_rate


def spks2neurometric(task_info, spk_mon, raster=False):
    """Calculates spike statistics from SpikeMonitor.spike_times(), following Naud & Sprekeler 2018."""
    from scipy.sparse import lil_matrix

    # params
    new_dt = unitless(task_info['sim']['stim_dt'], second, as_int=False)
    runtime = unitless(task_info['sim']['runtime'], second)
    settle_time = unitless(task_info['sim']['settle_time'], second)
    smooth_win = unitless(task_info['sim']['smooth_win'], second, as_int=False) / 2
    valid_burst = task_info['sim']['valid_burst']
    tps = unitless(int((runtime - settle_time)), new_dt)
    spk_times = spk_mon.spike_trains()
    nn = spk_times.__len__()

    # allocate variables
    events = lil_matrix((nn, tps), dtype='float32')   # events
    bursts = lil_matrix((nn, tps), dtype='float32')   # bursts
    singles = lil_matrix((nn, tps), dtype='float32')  # single spikes
    spikes = lil_matrix((nn, tps), dtype='float32')   # normal, all spikes
    all_isis = np.zeros(1)

    for n in np.arange(nn, dtype=int):
        this_spks = unitless(spk_times[n], second, as_int=False)
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
            eventtimes = this_spks[is_event.astype(bool)]
            bursttimes = this_spks[is_burst.astype(bool)]
            singltimes = this_spks[issingle.astype(bool)]

            # fill sparse matrix with the proper indices of the newdt
            events[n, handle_downsampled_spks(np.floor(eventtimes / new_dt)).astype(int)] = 1
            bursts[n, handle_downsampled_spks(np.floor(bursttimes / new_dt)).astype(int)] = 1
            singles[n, handle_downsampled_spks(np.floor(singltimes / new_dt)).astype(int)] = 1
            spikes[n, handle_downsampled_spks(np.floor(this_spks / new_dt)).astype(int)] = 1

    # sanity check
    all_isis *= 1e3      # in ms
    num_spikes = spikes.toarray().sum()
    assert len(spk_mon.t_[spk_mon.t_ >= settle_time]) == num_spikes, "You lost some spks while counting them..."

    if raster:
        return events.toarray(), bursts.toarray(), singles.toarray(), spikes.toarray(), all_isis

    # from lil_matrices2popraster and then 2rate per subpopulation
    sub = int(nn / 2)
    rates = []
    for i, matrix in enumerate([events, bursts, singles, spikes]):
        rate1, rate2 = smooth_rate(matrix.toarray(), smooth_win, new_dt, sub)
        rates.append(np.vstack((rate1, rate2)))

    # unpack accordingly
    eventrate = rates[0]
    burstrate = rates[1]
    singlerate = rates[2]
    firingrate = rates[3]

    return eventrate, burstrate, singlerate, firingrate, all_isis


# TODO: save burst times and spksperburst or not?


# def calculate_bursts(task_info):
#     dt = spksSE.clock.dt
#     validburst = task_info['sen']['2c']['validburst']
#     smooth_win_ = smooth_win / second
#
#     if task_info['sim']['burstanalysis']:
#
#         if task_info['sim']['2c_model']:
#             last_muOUd = np.array(dend_mon.muOUd[:, -int(1e3):].mean(axis=1))
#
#         if task_info['sim']['plasticdend']:
#             # calculate neurometric info per population
#             events, bursts, singles, spikes, isis = spks2neurometric(spksSE, runtime, settle_time, validburst,
#                                                                      smooth_win=smooth_win_, raster=False)
#
#             # plot & save weigths after convergence
#             eta0 = task_info['sen']['2c']['eta0']
#             tauB = task_info['sen']['2c']['tauB']
#             targetB = task_info['targetB']
#             B0 = tauB * targetB
#             tau_update = task_info['sen']['2c']['tau_update']
#             eta = eta0 * tau_update / tauB
#             plot_weights(dend_mon, events, bursts, spikes, [targetB, B0, eta, tauB, tau_update, smooth_win_], taskdir)
#             plot_rasters(spksSE, bursts, targetB, isis, runtime_, taskdir)
#         else:
#             # calculate neurometric per neuron
#             events, bursts, singles, spikes, isis = spks2neurometric(spksSE, runtime, settle_time, validburst,
#                                                                      smooth_win=smooth_win_, raster=True)
#             plot_neurometric(events, bursts, spikes, stim1, stim2, stimtime,
#                              (settle_time_, runtime_), taskdir, smooth_win_)
#             plot_isis(isis, bursts, events, (settle_time_, runtime_), taskdir)
