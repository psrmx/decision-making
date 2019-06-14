import numpy as np
from snep.utils import experiment_opener, filter_tasks
from helper_funcs import get_this_time, get_winner_loser_trials, instant_rate, choice_probability, pair_noise_corr, \
    plot_pop_averages, plot_fig2
from tqdm import tqdm
import pickle

load_path = '/Users/PSR/Documents/WS19/MasterThesis/Experiments/run_hierarchical'
test_expers = ['2019-06-14-19h09m14s_naud_-50dend_adaptation']
# test_expers = ['2019-06-04-16h47m04s']
target_var = 'bfb'
target_value = 10
plt_show = True
fig_extension = '.png'

# analysis params
compute_corr = False
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
    from brian2.units import ms

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
        is_winner_pop = np.zeros(n_trials, dtype=bool)
        pops, tps1 = tables.get_raw_data(target_ids[0], 'rates_dec').shape
        rates_dec = np.empty((n_trials, pops, tps1), dtype=np.float32)
        rates_sen = np.empty((n_trials, pops, tps1), dtype=np.float32)
        nn, tps2 = tables.get_computed(target_ids[0], 'spikes').shape
        spikes_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        bursts_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        events_av_per_trial = np.empty((int(2*nn), tps2), dtype=np.float32)
        tps_cp = int(tps2/step_cp)
        cp_av_per_trial = np.empty((nn, tps_cp), dtype=np.float32)
        e_cp_av_per_trial = np.empty((nn, tps_cp), dtype=np.float32)
        bf_cp_av_per_trial = np.empty((nn, tps_cp), dtype=np.float32)
        sub = int(nn / 2)
        corr = np.empty((nn, nn, int(tps_cp)), dtype=np.float32)
        rates_js = np.empty((nn, n_trials, tps_cp), dtype=np.float32)
        tps_stim = tables.get_raw_data(target_ids[0], 'stim_fluc').shape[0]
        stim_fluc = np.empty((n_trials, tps_stim), dtype=np.float32)

        for n in tqdm(range(nn)):
            this_n_spikes = np.empty((n_trials, tps2), dtype=np.float32)
            this_n_bursts = np.empty((n_trials, tps2), dtype=np.float32)
            this_n_events = np.empty((n_trials, tps2), dtype=np.float32)

            for i, tid in enumerate(target_ids):
                if n == 0:
                    rates_dec[i] = tables.get_raw_data(tid, 'rates_dec')
                    rates_sen[i] = tables.get_raw_data(tid, 'rates_sen')
                    stim_fluc[i] = tables.get_raw_data(tid, 'stim_fluc')
                    winner_pop = tables.get_raw_data(tid, 'winner_pop')[0]
                    if np.isclose(winner_pop, 0):
                        is_winner_pop[i] = True

                    if compute_corr:
                        for nj in range(nn):
                            rates_js[nj, i] = instant_rate(params, tables.get_computed(tid, 'spikes')[nj],
                                                           smooth_win=0.25)

                # retrieve spikes per neuron
                this_n_spikes[i] = tables.get_computed(tid, 'spikes')[n]
                this_n_bursts[i] = tables.get_computed(tid, 'bursts')[n]
                this_n_events[i] = tables.get_computed(tid, 'events')[n]

            # re-order winner pop
            if np.isclose(n,  sub):
                is_winner_pop = np.logical_not(is_winner_pop)
            spikes1, spikes2 = get_winner_loser_trials(this_n_spikes, is_winner_pop)
            bursts1, bursts2 = get_winner_loser_trials(this_n_bursts, is_winner_pop)
            events1, events2 = get_winner_loser_trials(this_n_events, is_winner_pop)

            # fill average arrays
            spikes_av_per_trial[n] = spikes1.mean(axis=0)
            spikes_av_per_trial[nn+n] = spikes2.mean(axis=0)
            bursts_av_per_trial[n] = bursts1.mean(axis=0)
            bursts_av_per_trial[nn+n] = bursts2.mean(axis=0)
            events_av_per_trial[n] = events1.mean(axis=0)
            events_av_per_trial[nn+n] = events2.mean(axis=0)

            # instantaneous rates and cps
            rates1 = instant_rate(params, spikes1, smooth_win=0.1, step=step_cp)
            rates2 = instant_rate(params, spikes2, smooth_win=0.1, step=step_cp)
            cp_av_per_trial[n] = choice_probability(rates1, rates2)
            e_rates1 = instant_rate(params, events1, smooth_win=0.1, step=step_cp)
            e_rates2 = instant_rate(params, events2, smooth_win=0.1, step=step_cp)
            e_cp_av_per_trial[n] = choice_probability(e_rates1, e_rates2)
            b_rates1 = instant_rate(params, bursts1, smooth_win=0.1, step=step_cp)
            b_rates2 = instant_rate(params, bursts2, smooth_win=0.1, step=step_cp)
            bf1 = b_rates1 / e_rates1
            bf1[np.isnan(bf1)] = 0  # handle division by zero
            bf2 = b_rates2 / e_rates2
            bf2[np.isnan(bf2)] = 0  # handle division by zero
            bf_cp_av_per_trial[n] = choice_probability(bf1, bf2)
            all_cps = [cp_av_per_trial, e_cp_av_per_trial, bf_cp_av_per_trial]

            # correlations
            if compute_corr:
                for nj in range(nn):
                    corr[n, nj, :] = pair_noise_corr(rates_js[n], rates_js[nj])

        # stim
        stim1, stim2 = get_winner_loser_trials(stim_fluc, np.logical_not(is_winner_pop))
        stim_diff = stim1.mean(axis=0) - stim2.mean(axis=0)
        stim_time = get_this_time(params, tps_stim, include_settle_time=True)

        # figures
        corr_ii = np.concatenate((corr[:sub, :sub], corr[sub:, sub:]), axis=0).reshape(-1, tps_cp)
        corr_ij = np.concatenate((corr[:sub, sub:], corr[sub:, :sub]), axis=0).reshape(-1, tps_cp)
        plot_pop_averages(params, rates_dec, rates_sen, all_cps, corr_ii, corr_ij, task_dir, '/fig1_'+fig_name)
        params['sim']['smooth_win'] = 100*ms
        plot_fig2(params, events_av_per_trial, bursts_av_per_trial, spikes_av_per_trial,
                  stim_diff, stim_time, rates_dec.mean(axis=0), False, task_dir, '/fig2_'+fig_name)

        # save variables
        file_name = task_dir + '/CPs_' + fig_name.replace(fig_extension, '.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump([rates_dec.mean(axis=0), rates_sen.mean(axis=0), cp_av_per_trial.mean(axis=0),
                         np.nanmean(corr_ii, axis=0), np.nanmean(corr_ij, axis=0),
                         events_av_per_trial, bursts_av_per_trial, spikes_av_per_trial], f)


if __name__ == '__main__':
    get_average_trials()
