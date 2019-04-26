import numpy as np
from snep.utils import experiment_opener, filter_tasks
from helper_funcs import plot_pop_averages, plot_fig2, choice_probability, get_winner_loser_trials
from circuits import get_mean_stim
from tqdm import tqdm

load_path = '/Users/PSR/Documents/WS19/MasterThesis/Experiments/run_hierarchical'
test_expers = ['2019-04-26-15h29m06s', '2019-04-26-15h29m18s']
target_var = 'bfb'
target_value = 1

plt_show = True
fig_extension = '.png'


@experiment_opener({'test_wimmer':  test_expers[0],
                    'test_naud':  test_expers[1],
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

        # params and allocate variables
        n_trials = len(target_ids)
        pops, tps1 = tables.get_raw_data(target_ids[0], 'rates_dec').shape
        rates_dec = np.empty((n_trials, pops, tps1))
        rates_sen = np.empty((n_trials, pops, tps1))
        nn, tps2 = tables.get_computed(target_ids[0], 'spikes').shape
        sub = int(nn/2)
        step = params['sim']['cp_step']
        spikes_av_per_trial = np.empty((int(2*nn), tps2))
        bursts_av_per_trial = np.empty((int(2*nn), tps2))
        events_av_per_trial = np.empty((int(2*nn), tps2))
        cp_av_per_trial = np.empty((nn, tps2-step))

        for n in tqdm(range(nn)):
            this_n_spikes = np.empty((n_trials, tps2))
            this_n_bursts = np.empty((n_trials, tps2))
            this_n_events = np.empty((n_trials, tps2))
            is_winner_pop = np.zeros(n_trials, dtype=bool)

            for i, tid in enumerate(target_ids):
                if n == 0:
                    rates_dec[i] = tables.get_raw_data(tid, 'rates_dec')
                    rates_sen[i] = tables.get_raw_data(tid, 'rates_sen')

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
            spikes_av_per_trial[int(2*sub)+n] = loser_spikes.mean(axis=0)
            bursts_av_per_trial[n] = winner_bursts.mean(axis=0)
            bursts_av_per_trial[int(2*sub)+n] = loser_bursts.mean(axis=0)
            events_av_per_trial[n] = winner_events.mean(axis=0)
            events_av_per_trial[int(2*sub)+n] = loser_events.mean(axis=0)
            cp_av_per_trial[n] = choice_probability(winner_spikes, loser_spikes, step=step)

        # figures
        mean_stim, stim_time = get_mean_stim(params, tps2)
        plot_pop_averages(params, rates_dec, rates_sen, cp_av_per_trial, task_dir, '/fig1_'+fig_name)
        plot_fig2(params, events_av_per_trial, bursts_av_per_trial, spikes_av_per_trial,
                  mean_stim, mean_stim, stim_time, task_dir, '/fig2_'+fig_name)


if __name__ == '__main__':
    get_average_trials()
    # explore_data()

# TODO: coherence levels and accuracy!
# TODO: correlations (?)
