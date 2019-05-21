import numpy as np
from snep.utils import experiment_opener, filter_tasks
from helper_funcs import plot_pop_averages, plot_fig2, get_winner_loser_trials, instantaneous_rate, choice_probability, pair_noise_corr
from circuits import get_mean_stim
from tqdm import tqdm
import pickle

load_path = '/Users/PSR/Documents/WS19/MasterThesis/Experiments/run_hierarchical'
# test_expers = ['2019-05-06-13h06m40s-50_trls-bfb_0_1_3',  '2019-05-14-17h24m15s']
test_expers = ['2019-05-18-13h11m07s'] #, ''2019-05-14-17h57m19s']
target_var = 'bfb'
target_value = 0
plt_show = True
fig_extension = '.png'

# analysis params
compute_corr = True
step_cp = 10


@experiment_opener({# 'test_wimmer':  test_expers[0],
                    'test_naud':  test_expers[0],
                    }, load_path, show=plt_show)
def get_average_trials(tables_task_ids):
    """
    Using the experiment_opener decorator automates some of the tedious aspects of handling experiment
    files, including opening and closing the file, plus it also calls plt.show() if you ask it to.
    And finally, it fixes a problem with SVG files so that they don't explode Inkscape if you import them.

    :param tables_task_ids: dict mapping from user supplied name to a tuple of (tables, task_ids)
    :return:
    """
    from snep.tables.experiment import ExperimentTables

    for t, test in enumerate(tables_task_ids):
        tables, task_ids = tables_task_ids[test]
        assert isinstance(tables, ExperimentTables)  # This allows PyCharm to autocomplete method names for tables
        task_dir = load_path + '/' + str(test)
        fig_name = test_expers[t] + '_' + target_var + str(target_value) + fig_extension
        params = tables.get_general_params(True)
        param_ranges = tables.read_param_ranges()

        # filter tasks to only the targets
        targets = [{('c',): 0, (target_var,): target_value}, ]
        target_ids = filter_tasks(task_ids, targets)
        params[target_var] = target_value

        # params and allocate variables
        n_trials = len(target_ids)
        pops, tps1 = tables.get_raw_data(target_ids[0], 'rates_dec').shape
        rates_dec = np.empty((n_trials, pops, tps1), dtype=np.float32)
        rates_sen = np.empty((n_trials, pops, tps1), dtype=np.float32)
        nn, tps2 = tables.get_computed(target_ids[0], 'spikes').shape
        spikes_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        bursts_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        events_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        tps_cp = int(tps2/step_cp)
        cp_av_per_trial = np.empty((nn, tps_cp), dtype=np.float32)
        sub = int(nn / 2)
        sub_corr = int(sub / 2)
        nn_corr = np.hstack((np.arange(sub_corr), np.arange(nn - sub_corr, nn))).astype(dtype=np.int16)
        corr = np.empty((sub, sub, tps_cp), dtype=np.float32)
        rates_js = np.empty((sub, n_trials, tps_cp), dtype=np.float32)

        for n in tqdm(range(nn)):
            this_n_spikes = np.empty((n_trials, tps2), dtype=np.float32)
            this_n_bursts = np.empty((n_trials, tps2), dtype=np.float32)
            this_n_events = np.empty((n_trials, tps2), dtype=np.float32)
            is_winner_pop = np.zeros(n_trials, dtype=bool)

            for i, tid in enumerate(target_ids):
                if n == 0:
                    rates_dec[i] = tables.get_raw_data(tid, 'rates_dec')
                    rates_sen[i] = tables.get_raw_data(tid, 'rates_sen')

                    if compute_corr:
                        for j, nj in enumerate(nn_corr):
                            rates_js[j, i] = instantaneous_rate(params, tables.get_computed(tid, 'spikes')[nj],
                                                                smooth_win=0.25)
                # retrieve spikes per neuron
                this_n_spikes[i] = tables.get_computed(tid, 'spikes')[n]
                this_n_bursts[i] = tables.get_computed(tid, 'bursts')[n]
                this_n_events[i] = tables.get_computed(tid, 'events')[n]
                winner_pop = tables.get_raw_data(tid, 'winner_pop')[0]
                if winner_pop == np.floor(n/sub):
                    is_winner_pop[i] = True

            # re-order winner pop
            winner_spikes, loser_spikes = get_winner_loser_trials(this_n_spikes, is_winner_pop)
            winner_bursts, loser_bursts = get_winner_loser_trials(this_n_bursts, is_winner_pop)
            winner_events, loser_events = get_winner_loser_trials(this_n_events, is_winner_pop)

            # fill average arrays
            spikes_av_per_trial[n] = winner_spikes.mean(axis=0)
            spikes_av_per_trial[nn+n] = loser_spikes.mean(axis=0)
            bursts_av_per_trial[n] = winner_bursts.mean(axis=0)
            bursts_av_per_trial[nn+n] = loser_bursts.mean(axis=0)
            events_av_per_trial[n] = winner_events.mean(axis=0)
            events_av_per_trial[nn+n] = loser_events.mean(axis=0)

            # instantaneous rates and cps
            winner_rates = instantaneous_rate(params, winner_spikes, smooth_win=0.1, step=step_cp)
            loser_rates = instantaneous_rate(params, loser_spikes, smooth_win=0.1, step=step_cp)
            cp_av_per_trial[n] = choice_probability(winner_rates, loser_rates)

            # correlations
            if compute_corr and (n < sub_corr or nn - sub_corr <= n < nn):
                n_idx = n
                if nn - sub_corr <= n < nn:
                    n_idx = n - sub
                for j in range(sub):
                    corr[n_idx, j, :] = pair_noise_corr(rates_js[n_idx], rates_js[j])

        # figures
        corr_ii = np.concatenate((corr[:sub_corr, :sub_corr], corr[sub_corr:, sub_corr:]), axis=0).reshape(-1, tps_cp)
        corr_ij = np.concatenate((corr[:sub_corr, sub_corr:], corr[sub_corr:, :sub_corr]), axis=0).reshape(-1, tps_cp)
        mean_stim, stim_time = get_mean_stim(params, tps2)
        plot_pop_averages(params, rates_dec, rates_sen, cp_av_per_trial, corr_ii, corr_ij, task_dir, '/fig1_'+fig_name)
        plot_fig2(params, events_av_per_trial, bursts_av_per_trial, spikes_av_per_trial,
                  mean_stim, stim_time, rates_dec.mean(axis=0), False, task_dir, '/fig2_'+fig_name)

        # save variables
        file_name = task_dir + '/CPs_' + fig_name.replace(fig_extension, '.pkl')
        with open(file_name, 'wb') as f:
                pickle.dump([rates_dec.mean(axis=0), rates_sen.mean(axis=0), cp_av_per_trial.mean(axis=0),
                             np.nanmean(corr_ii, axis=0), np.nanmean(corr_ij, axis=0),
                             events_av_per_trial, bursts_av_per_trial, spikes_av_per_trial], f)


if __name__ == '__main__':
    get_average_trials()
