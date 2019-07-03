from snep.utils import experiment_opener, filter_tasks
from helper_funcs import plot_isis, np_array
from tqdm import tqdm
import pickle

load_path = '/Users/PSR/Documents/WS19/MasterThesis/Experiments/run_hierarchical'
test_expers = ['2019-07-01-23h04m29s-naud_tuned_single_trl']
target_var = 'bfb'
target_value = 0
plt_show = True
fig_extension = '.png'


@experiment_opener({# 'test_wimmer':  test_expers[0],
                    'test_naud':  test_expers[0],
                    }, load_path, show=plt_show)
def get_isis(tables_task_ids):
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
        fig_name = '/fig_isis' + test_expers[t] + '_' + target_var + str(target_value) + fig_extension
        params = tables.get_general_params(True)
        param_ranges = tables.read_param_ranges()

        # filter tasks to only the targets
        targets = [{('c',): 0, (target_var,): target_value}, ]
        target_ids = filter_tasks(task_ids, targets)
        params[target_var] = target_value

        # allocate variables to save
        cvs = []
        isis = []
        ieis = []
        ibis = []
        spks_per_burst = []

        for i, tid in tqdm(enumerate(target_ids)):
            cvs.extend(tables.get_computed(tid, 'cvs'))
            isis.extend(tables.get_computed(tid, 'isis'))
            ieis.extend(tables.get_computed(tid, 'ieis'))
            ibis.extend(tables.get_computed(tid, 'ibis'))
            spks_per_burst.extend(tables.get_computed(tid, 'spks_per_burst'))

        # from lists to arrays
        cvs = np_array(cvs)
        isis = np_array(isis)
        ieis = np_array(ieis)
        ibis = np_array(ibis)
        spks_per_burst = np_array(spks_per_burst)

        # plot figure and save data
        plot_isis(params, cvs, isis, ieis, ibis, spks_per_burst, task_dir, fig_name)
        file_name = task_dir + fig_name.replace(fig_extension, '.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump([cvs, isis, ieis, ibis, spks_per_burst], f)

@experiment_opener({# 'test_wimmer':  test_expers[0],
                    'test_naud':  test_expers[0],
                    }, load_path, show=plt_show)
def get_single_trial(tables_task_ids):
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
        fig_name = '/fig_sinlge_trial' + test_expers[t] + '_' + target_var + str(target_value) + fig_extension
        params = tables.get_general_params(True)
        param_ranges = tables.read_param_ranges()

        # filter tasks to only the targets
        targets = [{('c',): 0, (target_var,): target_value, ('iter',): 0}]
        target_ids = filter_tasks(task_ids, targets)
        params[target_var] = target_value

        for i, tid in tqdm(enumerate(target_ids)):
            spksi_DE = tables.get_sim_state(tid, 'spksi_dec')
            spkst_DE = tables.get_sim_state(tid, 'spkst_dec')
            spksi_SE = tables.get_sim_state(tid, 'spksi_sen')
            spkst_SE = tables.get_sim_state(tid, 'spkst_sen')
            rate_t = tables.get_sim_state(tid, 'rate_t')
            rate_DE1 = tables.get_sim_state(tid, 'rate_DE1')
            rate_DE2 = tables.get_sim_state(tid, 'rate_DE2')
            rate_DI = tables.get_sim_state(tid, 'rate_DI')
            rate_SE1 = tables.get_sim_state(tid, 'rate_SE1')
            rate_SE2 = tables.get_sim_state(tid, 'rate_SE2')
            rate_SI = tables.get_sim_state(tid, 'rate_SI')
            stim1 = tables.get_sim_state(tid, 'stim1')
            stim2 = tables.get_sim_state(tid, 'stim2')
            stim_time = tables.get_sim_state(tid, 'stim_time')

        # plot figure and save data
        file_name = task_dir + fig_name.replace(fig_extension, '.pkl')
        with open(file_name, 'wb') as f:
            pickle.dump([spksi_DE, spkst_DE, spksi_SE, spkst_SE, rate_t,
                         rate_DE1, rate_DE2, rate_DI, rate_SE1, rate_SE2, rate_SI,
                         stim1, stim2, stim_time], f)


if __name__ == '__main__':
    # get_isis()
    get_single_trial()
