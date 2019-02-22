import numpy as np
from brian2.units import *


def spks2neurometric(spkmon, runtime, settletime, validburst=16e-3, smoothwin=50e-3, step=10, raster=False):
    """
    Calculate spikes statistics from Brian simulations,
    according to Naud & Sprekeler 2018 and Filip's code.

    :param spkmon: SpikeMonitor object
    :param runtime: total simulation time, in seconds.
    :param validburst: a float, maximum time interval between spks to count as bursts, in seconds.
    :param smoothwin: a float, width of the rectangular window to smooth.
    :param step: a float, downsample step size.
    :param raster: a bool, whether to output raster matrices or not.
    :return:
    """
    from scipy.sparse import lil_matrix

    # monitor params
    dt = spkmon.clock.dt_
    spktimes = spkmon.spike_trains()
    if not runtime.is_dimensionless:
        runtime /= second
    if not settletime.is_dimensionless:
        settletime /= second

    # network params
    NE = spktimes.__len__()
    timepts = int((runtime - settletime) / dt)

    # params for smoothing and donwsampling
    newtimepts = int(timepts / step)  # step=10 for output of size 3000
    newdt = dt*step
    kernel = np.ones((int(smoothwin / newdt)))

    # lil matrices are efficient in constructing rasters as sparse matrices, notice that dt=1ms (downsampling)
    events = lil_matrix((NE, newtimepts), dtype='float32')  # events
    bursts = lil_matrix((NE, newtimepts), dtype='float32')  # bursts
    singles = lil_matrix((NE, newtimepts), dtype='float32')  # single spikes
    spikes = lil_matrix((NE, newtimepts), dtype='float32')  # normal, all spikes
    allisis = np.empty(0)

    # loop through neurons and count bursts!
    for n in np.arange(NE, dtype=int):
        # get spktimes of this neuron, unitless
        thisspks = spktimes[n] / second
        thisspks = thisspks[thisspks >= settletime]  # filter to ignore spks during settle time
        thisspks -= settletime

        if len(thisspks) > 0:
            # count as burst if the following spike is within 16 ms apart from the previous one
            # Initial False compensates for the diff loss, and the case of one spk
            isis = np.diff(thisspks)
            allisis = np.hstack((allisis, isis))
            thisbursts = np.concatenate(([False], isis < validburst)).astype(bool)

            # get events as ~thisbursts, because thisbursts output True on the second spk!
            isburst = thisbursts.astype('float32')
            isevent = np.array(~thisbursts).astype('float32')
            spksperburst = np.zeros(isburst.shape)
            numbursts = 0

            # if there are bursts!
            if isburst.any():
                # add preceding burst
                begbursts = np.where(np.diff(isburst) == True)[0]
                numbursts = len(begbursts)
                isburst[begbursts] = 1

                # count number of spikes in each burst
                ibi = np.concatenate((begbursts, isburst.shape))  # inter-burst-intervals markers
                nspksperburst = np.array([isburst[ibi[b]:ibi[b + 1]].sum() for b in range(numbursts)])
                nspksperburst[nspksperburst == 1] = 2  # sanity check, there's no 1 spk burst!
                isburst[thisbursts] = 0  # rmv consecutive burst spks
                spksperburst[isburst.astype(bool)] = nspksperburst

            # get single spks
            issingle = np.logical_and(isburst == 0, isevent.astype(bool))

            # sanity check
            allspks = isburst.sum() + issingle.sum() + spksperburst.sum() - numbursts
            assert allspks == len(thisspks), "ups, sth is weird in the burst quantification :("

            # get events, bursts, singles times
            eventtimes = thisspks[isevent.astype(bool)]
            bursttimes = thisspks[isburst.astype(bool)]
            singltimes = thisspks[issingle.astype(bool)]

            # helper func2create mask with proper indices
            def check4twospks(mask):
                conflict = np.where(np.diff(mask) == 0)[0]
                mask[conflict] -= 1
                # recursive search for no 2spks in a newdt bin
                if np.any(np.diff(mask) == 0):
                    mask = check4twospks(mask)
                return mask

            # fill sparse matrix with the proper indices of the newdt
            events[n, check4twospks(np.floor(eventtimes / newdt)).astype(int)] = 1
            bursts[n, check4twospks(np.floor(bursttimes / newdt)).astype(int)] = 1
            singles[n, check4twospks(np.floor(singltimes / newdt)).astype(int)] = 1
            spikes[n, check4twospks(np.floor(thisspks / newdt)).astype(int)] = 1

    # sanity check
    allisis *= 1e3      # in ms
    assert spkmon.t_[spkmon.t_ >= settletime].__len__() == spikes.toarray().sum(), "You lost some spks on the way of counting them..."

    if raster:
        return events.toarray(), bursts.toarray(), singles.toarray(), spikes.toarray(), allisis

    # from lil_matrices2popraster and then 2rate per subpopulation
    sub = int(NE / 2)
    rates = []
    for i, matrix in enumerate([events, bursts, singles, spikes]):
        popraster1 = np.squeeze(matrix.toarray()[:sub].mean(axis=0)) / smoothwin
        popraster2 = np.squeeze(matrix.toarray()[sub:].mean(axis=0)) / smoothwin
        rate1 = np.convolve(popraster1, kernel, mode='same')
        rate2 = np.convolve(popraster2, kernel, mode='same')
        rates.append(np.vstack((rate1, rate2)))

    # unpack accordingly
    eventrate = rates[0]
    burstrate = rates[1]
    singlerate = rates[2]
    firingrate = rates[3]

    return eventrate, burstrate, singlerate, firingrate, allisis


# TODO: save burst times and spksperburst or not?


def spks2neurometrictimes(spktimes, dt, runtime, validburst=16e-3):
    """
    Calculate spikes statistics from Brian simulations,
    according to Naud & Sprekeler 2018 and Filip's code.

    :param spktimes: a SpikeMonitor object or a dict from the SpikeMonitor.
    :param dt: integration step of Monitor, in seconds.
    :param runtime: total simulation time, in seconds.
    :param validburst: a float, maximum time interval between spks to count as busts, in seconds.
    :return: etimes, btimes, stimes
    """
    # remove units from the params
    try:
        spktimes = spktimes.spike_trains()
    except AttributeError:
        pass
    for i, spks in spktimes.items():
        if type(spks) != np.ndarray:
            spktimes[i] = spks / second
    if type(runtime) != float:
        runtime /= second
    if type(dt) != float:
        dt /= second

    # get important params and initialise matrices
    NE = spktimes.__len__()
    evnttimes = spktimes.copy()  # events
    brsttimes = spktimes.copy()  # bursts
    sngltimes = spktimes.copy()  # singles

    # loop through neurons and count bursts!
    for n in np.arange(NE, dtype=int):
        thisspks = spktimes[n]
        thisspks = thisspks[thisspks >= 0.5]  # filter to ignore spks during settle time

        if len(thisspks) > 0:
            # count as burst if the following spike is within 16 ms apart from the previous one
            # Initial False compensates for the diff loss, and the case of one spk
            isis = np.diff(thisspks)
            thisbursts = np.concatenate(([False], isis < validburst)).astype(bool)

            # get events as ~thisbursts, because thisbursts output True on the second spk!
            isburst = thisbursts.astype('float32')
            isevent = np.array(~thisbursts).astype('float32')
            spksperburst = np.zeros(isburst.shape)
            numbursts = 0

            # if there are bursts!
            if isburst.any():
                # add preceding burst
                begbursts = np.where(np.diff(isburst) == True)[0]
                numbursts = len(begbursts)
                isburst[begbursts] = 1

                # count number of spikes in each burst
                ibi = np.concatenate((begbursts, isburst.shape))  # inter-burst-intervals markers
                nspksperburst = np.array([isburst[ibi[b]:ibi[b + 1]].sum() for b in range(numbursts)])
                nspksperburst[nspksperburst == 1] = 2  # sanity check, there's no 1 spk burst!
                isburst[thisbursts] = 0  # rmv consecutive burst spks
                spksperburst[isburst.astype(bool)] = nspksperburst

            # get single spks
            issingle = np.logical_and(isburst == 0, isevent.astype(bool))

            # sanity check
            allspks = isburst.sum() + issingle.sum() + spksperburst.sum() - numbursts
            assert allspks == len(thisspks), "ups, sth is weird in the burst quantification :("

            # get events, bursts, singles times
            evnttimes[n] = thisspks[isevent.astype(bool)]
            brsttimes[n] = thisspks[isburst.astype(bool)]
            sngltimes[n] = thisspks[issingle.astype(bool)]

    return evnttimes, brsttimes, sngltimes, spktimes
