import tables
import warnings
from .network import NetworkTables
from .results import ResultsTables
from .paramspace import ParameterSpaceTables
import os
from collections import namedtuple
# from .lock import LockAllFunctionsMeta
# from six import with_metaclass
warnings.simplefilter('ignore', tables.NaturalNameWarning)


class ExperimentTables(NetworkTables, ResultsTables, ParameterSpaceTables):
    # class ExperimentTables(with_metaclass(LockAllFunctionsMeta, (NetworkTables,ResultsTables,ParameterSpaceTables))):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    This is the master hdf5 file handler class. It is responsible for creating an experiment
    file, in which network configuration and data should be stored. It inherits from NetworkTables
    and ResultsTables which do no file handling, but can read and write subtrees from the
    master experiment file.
    '''
    def __init__(self, filename, flat_results_groups=True):
        '''
        filename - must be a complete path, including the hdf5 file name
        '''
        self.filename = filename

        ParameterSpaceTables.__init__(self, flat_results_groups)
        ResultsTables.__init__(self, ExperimentTables.log_info, ExperimentTables.log_err)
        NetworkTables.__init__(self)

    def __del__(self):
        self.close_file()

    @staticmethod
    def log_info(msg, h5f=None):
        print(msg)

    @staticmethod
    def log_err(msg, h5f=None):
        print(msg)

    def open_file(self, readonly=False):
        from snep.tables.data import DataHandler
        if readonly:
            self.h5f = tables.open_file(self.filename, mode="r")
            ResultsTables.set_root(self, self.h5f.root.results)
            NetworkTables.set_root(self, self.h5f.root.network)
            ParameterSpaceTables.set_root(self, self.h5f.root.paramspace)
        else:
            self.h5f = tables.open_file(self.filename, mode="a",
                                        title="Master Experiment File")
        self.handler = DataHandler(self.h5f, self.log_info, self.log_err)

    def close_file(self):
        if self.h5f and self.h5f.isopen:
            self.h5f.close()

    def initialize(self):
        '''
        Once a new file is opened, it should be populated with the appropriate default
        groups and tables. This class only needs to create the experiment group at the root
        of the file. It then passes that group as the parent group for the NetworkTables
        and ResultsTables, where they will construct their own groups and tables.
        '''
        ParameterSpaceTables.initialize(self,self.h5f, self.h5f.root)
        NetworkTables.initialize(self, self.h5f, self.h5f.root)
        ResultsTables.initialize(self, self.h5f, self.h5f.root)

    def results_file(self, resultpath):
        '''
        This should probably be specified somehow in the configuration.py but for now
        it's here.
        '''
        return os.path.join(resultpath, 'results.h5')
    
    def get_task_info(self, task_id):
        return self.as_dictionary(task_id, True)
    
    def task_name(self, task_id):
        return self.get_results_group_path(task_id)

    def get_task_ids(self, onlyfinished=False, onlyunfinished=False):
        paramspace_pts = self.paramspace_pts(onlyunfinished=onlyunfinished, onlyfinished=onlyfinished)
        if not paramspace_pts:
            self.build_parameter_space()
            paramspace_pts = self.paramspace_pts()
        return paramspace_pts
    
    def copy_parameters(self, other, reset_results, new_seeds):
        other.copy_network(self)
        other.copy_paramspace(self)
        if reset_results:
            self.reset_results(new_seeds=new_seeds)

    def param_range_helpers(self, types, formats, filter_by=None, onlyfinished=False, onlyunfinished=False, defaults=None):
        '''
        :param types: An iterable containing the ids of all the param ranges to include
        :param formats: A dictionary from param ids to functions that can pretty-print the value
        :param filter_by: A dictionary from param ids to default values for any ranges that are not to be included,
                            UNLESS they are also in 'types' in which case we include them in ranges, but only that value
        :return:
        '''
        from snep.utils import filter_tasks, ParametersNamed, ParameterArray, ParameterSpace
        import numpy as np
        param_range = namedtuple('param_range', ['n', 'values', 'names', 'map'])
        task_ids = self.get_task_ids(onlyfinished, onlyunfinished)

        indexed_types = tuple(types)
        if filter_by is None:
            ignore_ranges = []
            target_ids = task_ids
            filter_set = set()
        else:
            ignore_ranges = list(filter_by.keys())
            target_ids = filter_tasks(task_ids, [filter_by])
            filter_set = set(filter_by.keys())

        param_ranges = self.read_param_ranges()

        if defaults is None:
            defaults = {}
        types = list(types)
        for t in list(defaults.keys()):
            # if t in param_ranges:
            #     print('{} found in defaults and in param ranges!'.format(t))
            #     del defaults[t]
            if t in types:
                types.remove(t)

        types_set = set(types)
        missing = set(param_ranges.keys()) - types_set - filter_set - set(defaults)
        param_links = ParameterSpace.transitive_set_reduce(self._read_param_links())
        all_linked = set()
        for pl in param_links:
            for p in pl:
                all_linked.add(p)
            pl_set = set(pl)
            matched_types = pl_set.intersection(types_set)
            matches_filter = pl_set.intersection(filter_set)
            # assert len(matches_filter) == 0 or len(matched_types) == 0
            if matched_types:
                linked_to = pl_set - types_set
                for m in linked_to:
                    missing.remove(m)
            if matches_filter:
                linked_to = pl_set - filter_set
                for m in linked_to:
                    missing.remove(m)
                    ignore_ranges.append(m)
        # for t in defaults:
        #     if t in missing:
        #         missing.remove(t)

        assert not missing, 'Did not specify {} in types or filter_by'.format(missing)

        range_attributes = {}
        for t in types:
            # if t in filter_by:
            #     # Means we should create length 1 arrays
            #     fb_v = filter_by[t]
            #     pr = param_ranges[t]
            #     if isinstance(pr, ParametersNamed):
            #         t_range = pr.names_values
            #         values = [v for n, v in t_range if v == fb_v]
            #         names = [n for n, v in t_range if v == fb_v]
            #         sorted_idx = np.argsort(values)
            #         values = [values[i] for i in sorted_idx]
            #         if t not in formats:
            #             names = [names[i] for i in sorted_idx]
            #         else:
            #             names = [formats[t](v) for v in values]
            #     elif isinstance(pr, ParameterArray):
            #         t_range = pr.value
            #         values = sorted(v for v in t_range)
            #         if t not in formats:
            #             names = [str(v) for v in values]
            #         else:
            #             names = [formats[t](v) for v in values]
            #     t_map = {v: i for i, v in enumerate(values)}
            #     n = len(values)
            #     range_attributes[t] = param_range(n, np.array(values), np.array(names), t_map)
            if t not in ignore_ranges:
                pr = param_ranges[t]
                if isinstance(pr, ParametersNamed):
                    t_range = pr.names_values
                    values = [v for n, v in t_range]
                    names = [n for n, v in t_range]
                    # sorted_idx = np.argsort(values)
                    # values = [values[i] for i in sorted_idx]
                    if t in formats:
                        names = [formats[t](v) for v in values]
                    #     names = [names[i] for i in sorted_idx]
                    # else:
                elif isinstance(pr, ParameterArray):
                    t_range = pr.value
                    values = [v for v in t_range]
                    if t not in formats:
                        names = [str(v) for v in values]
                    else:
                        names = [formats[t](v) for v in values]
                else:
                    raise TypeError('Cannot handle given type: {}'.format(type(pr)))
                n = len(values)
                if len(set(values)) != n:
                    t_map = {}
                else:
                    t_map = {v: i for i, v in enumerate(values)}
                range_attributes[t] = param_range(n, np.array(values), np.array(names), t_map)
        for t, v in defaults.items():
            n = formats[t](v) if t in formats else str(v)
            range_attributes[t] = param_range(1, np.array([v]), np.array([n]), {v: 0})
        # indexed_types = range_attributes.keys()

        all_values = {}
        for t in types + list(all_linked):
            if t in range_attributes:
                all_values[t] = range_attributes[t].values
            elif isinstance(param_ranges[t], ParametersNamed):
                all_values[t] = np.array([v for n, v in param_ranges[t].names_values])
            elif isinstance(param_ranges[t], ParameterArray):
                all_values[t] = np.array([v for v in param_ranges[t].value])
            else:
                assert False, 'Something is borked. Ask Owen to help!'

        def values(tid, t):
            if t in defaults:
                return defaults[t]
            else:
                return tid[t].value
        def do_map(tid):
            all_indices = {}
            for pl in param_links:
                matched_at = np.ones(len(all_values[pl[0]]), dtype=np.bool)
                for p in pl:
                    pi = values(tid, p) == all_values[p]
                    matched_at = np.logical_and(matched_at, pi)
                matched_i = np.argwhere(matched_at).flatten()[0]
                for p in pl:
                    all_indices[p] = matched_i
            for t in indexed_types:
                if t not in all_linked:
                    all_indices[t] = range_attributes[t].map[values(tid, t)]
            return tuple(all_indices[t] for t in indexed_types)


        def shape(prefix=(), postfix=()):
            prefix = prefix if isinstance(prefix, tuple) else (prefix,)
            postfix = postfix if isinstance(postfix, tuple) else (postfix,)
            return prefix + tuple(range_attributes[t].n for t in indexed_types) + postfix
        def idx(tid, prefix=(), postfix=()):
            prefix = prefix if isinstance(prefix, tuple) else (prefix,)
            postfix = postfix if isinstance(postfix, tuple) else (postfix,)
            return prefix + do_map(tid) + postfix
        def sv(tid, a, v, prefix=(), postfix=()):
            prefix = prefix if isinstance(prefix, tuple) else (prefix,)
            postfix = postfix if isinstance(postfix, tuple) else (postfix,)
            a[idx(tid, prefix, postfix)] = v

        return target_ids, range_attributes, idx, sv, shape

