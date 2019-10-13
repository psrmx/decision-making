from snep.utils import experiment_opener, filter_tasks
import numpy as np
import matplotlib.pyplot as plt

plt_show = True
load_path = '~/experiments/jobexp'


def explore_data():
    '''
    The simplest example of manually opening an experiment file and accessing various bits of data.
    :return: 
    '''
    from snep.utils import make_tables_from_path
    tables = make_tables_from_path(load_path)
    tables.open_file(True)
    task_ids = tables.get_task_ids(False)
    sleeps = np.empty(len(task_ids))
    for i, tid in enumerate(task_ids):
        sim_state = tables.get_sim_state(tid)
        computed = tables.get_computed(tid)
        raw_data = tables.get_raw_data(tid)
        sleeps[i] = sim_state['sleep']
        print(sim_state['sleep'], computed['sum'], raw_data['b'].shape)

    tables.close_file()
    plt.plot(sleeps)
    plt.show()


@experiment_opener({'test0':  '2017-05-12-12h19m26s',
                    # 'test1':  '2017-02-20-22h49m54s'
                    }, load_path, show=plt_show)
def plot_figures(tables_task_ids):
    '''
    Using the experiment_opener decorator automates some of the tedious aspects of handling experiment
    files, including opening and closing the file, plus it also calls plt.show() if you ask it to.
    And finally, it fixes a problem with SVG files so that they don't explode Inkscape if you import them.
    
    :param tables_task_ids: dict mapping from user supplied name to a tuple of (tables, task_ids)
    :return: 
    '''
    from snep.tables.experiment import ExperimentTables

    tables, task_ids = tables_task_ids['test0']
    assert isinstance(tables, ExperimentTables)  # This allows PyCharm to autocomplete method names for tables
    params = tables.get_general_params(True)
    param_ranges = tables.read_param_ranges()

    targets = [{('x', 'q', 'gl'): 9,
                # ('y', 'b'): 15.,
                },]
    target_ids = filter_tasks(task_ids, targets)

    x_type = ('sleep',)
    x_range = param_ranges[x_type].names_values
    x_values = [v for n, v in x_range]
    x_names = [n for n, v in x_range]
    x_map = {v: i for i, v in enumerate(x_values)}
    x_n = len(x_map)

    sleeps = np.empty(x_n)
    for tid in target_ids:
        task_params = tables.get_task_info(tid)
        sleep = tables.get_sim_state(tid, 'sleep')
        computed = tables.get_computed(tid)
        raw_data = tables.get_raw_data(tid)
        x_i = x_map[tid[x_type].value]
        sleeps[x_i] = sleep

    plt.plot(x_values, sleeps)
    # No need to call plt.show because we told the experiment_opener to do it for us.


if __name__ == '__main__':
    plot_figures()
    # explore_data()
