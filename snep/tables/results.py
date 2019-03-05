import tables
from scipy.sparse import csr_matrix
from snep.utils import csr_make_ints
from snep.tables.data import DataHandler

class ResultsTablesReader(object):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the simulation results from the experiment.
    '''

    '''
    These internal properties are defined so that we can restructure the hdf5 file without
    modifying the code that accesses it anywhere but here.
    '''
    _sim_state = property(fget=lambda self: self.results.sim_state)
    _raw_data = property(fget=lambda self: self.results.raw_data)
    _computed = property(fget=lambda self: self.results.computed)
    _logs = property(fget=lambda self: self.results.log_files)

    def __init__(self, log_info, log_err):
        self.h5f = None
        self.handler = None
        self.log_info = log_info
        self.log_err = log_err

    def set_root(self, results):
        ''' 
        results - Specifies the root group to read from.
        '''
        self.results = results

    def _build_full_path(self, paramspace_pt, path):
        if paramspace_pt is None:
            full_path = path
        elif path is None:
            full_path = self.get_results_group_path(paramspace_pt)
        else:
            psp_path = self.get_results_group_path(paramspace_pt)
            full_path = '/'.join((psp_path,path))
        return full_path
        
    def get_raw_data(self, paramspace_pt=None, path=None, key=None):
        full_path = self._build_full_path(paramspace_pt, path)
        node = self.h5f.get_node(self._raw_data, full_path)
        all_data = self.handler._read_node(node, key)
        return all_data
    
    def get_sim_state(self, paramspace_pt=None, path=None, key=None):
        full_path = self._build_full_path(paramspace_pt, path)
        node = self.h5f.get_node(self._sim_state, full_path)
        all_data = self.handler._read_node(node, key)
        return all_data
    
    def get_computed(self, paramspace_pt=None, path=None, key=None):
        full_path = self._build_full_path(paramspace_pt, path)
        node = self.h5f.get_node(self._computed, full_path)
        all_data = self.handler._read_node(node, key)
        return all_data
    
    # def _read_node(self, node, key):
    #     '''
    #     :param node:
    #     :param key: Tuple. Numpy-style fancy index for an array. e.g. (Ellipse, slice(3, -1, 2))
    #     :return:
    #     '''
    #     if isinstance(node, tables.Group):
    #         data = self._read_group(node)
    #     elif isinstance(node, tables.VLArray):
    #         data = self._read_VLArray(node)
    #     else:  # for tables.CArray and tables.Array
    #         if key is not None:
    #             data = node.__getitem__(key)
    #         else:
    #             data = node.read()
    #     return data
    #
    # def _read_group(self, group):
    #     if 'issparse' in group._v_attrs._f_list():
    #         data = self._read_sparse(group)
    #     else:
    #         data = {node._v_name:self._read_node(node, None)
    #                     for node in group._f_iter_nodes()}
    #     return data
    #
    # def _read_VLArray(self, vlarray, asdictionary=True):
    #     vla = vlarray.read()
    #     if asdictionary:
    #         data = {i:values for i,values in enumerate(vla)}
    #     else:
    #         data = [(i,v) for i,values in enumerate(vla) for v in values]
    #     return data
    #
    # def _read_sparse(self, group):
    #     data   = group.data.read()
    #     indices= group.indices.read()
    #     indptr = group.indptr.read()
    #     shape  = group.shape.read()
    #     indptr, indices = csr_make_ints(indptr, indices)
    #     arr = csr_matrix((data,indices,indptr),shape=shape)
    #     return arr

    def get_connection_weights(self, paramspace_pt, connection_name):
        """
        See snep.utils.WeightMonitor.convert_values for how the weights are recorded and stored.
        :param paramspace_pt:
        :type paramspace_pt: dict
        :param connection_name:
        :type connection_name: str
        :return:
        :rtype: dict
        """
        weights = self.get_raw_data(paramspace_pt, 'weights/'+connection_name)
        if 'all' in weights:
            all_w = weights.pop('all')
            all_data = all_w['all_data']
            indices = all_w['indices']
            indptr = all_w['indptr']
            shape = all_w['shape']
            weights['all'] = [csr_matrix((all_data[i, :], indices, indptr), shape=shape) for i in range(all_data.shape[0])]
        return weights


class ResultsTables(ResultsTablesReader):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to write a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the simulation results from the experiment.
    
    The two most important functions from a user perspective are add_computed
    and add_raw_data. They both store data using the same basic structure. Any
    data to be stored in the HDF5 file should be passed in the all_data 
    dictionary parameter. That dictionary will recursively map to structures 
    in the file as follows:
    - Any value in a dictionary which is a dense ndarray will be 
      stored as a CArray, whose name is the corresponding key.
    - Any value in a dictionary which is a sparse array will be stored
      as the data structures underlying the equivalent csr_matrix representation.
    - Any value in a dictionary which is another dictionary whose keys are all
      integers and values are all ndarrays will be stored as a VLArray.
    - Any value in a dictionary which is another dictionary whose keys are
      strings (or is empty) will be stored as a group, whose values will be
      defined as above.
      
    all_data = {
    'nameofdense':numpy.array,
    'nameofsparse':scipy.sparse,
    'nameofvlarray':{0:ndarray, 1:ndarray, ... n:ndarray},
    'nameofgroup':{'subgroup':{}, 'somearray':ndarray, etc.}
    }
     
    
    See those functions for further details on how to use them.
    '''
    def __init__(self, log_info, log_err):
        """
        The default compression has not really been tested, I don't know if this works as expected.
        """
        ResultsTablesReader.__init__(self, log_info, log_err)

    def initialize(self, h5f, parentgroup):
        """
        Once the ExperimentTables class has created a new hdf5 file it passes the root
        group into this function as the parentgroup. Here we then create the necessary
        default groups and tables that this class is responsible for.
        """
        try:
            results = parentgroup.results
        except tables.NoSuchNodeError:
            results =  h5f.create_group(parentgroup, 'results')
        ResultsTablesReader.set_root(self, results)
        
        try:
            self.results.log_files
        except tables.NoSuchNodeError:
            h5f.create_group(self.results, 'log_files')

        try:
            self.results.sim_state
        except tables.NoSuchNodeError:
            sim_state = h5f.create_group(self.results, 'sim_state')

        try:
            self.results.raw_data
        except tables.NoSuchNodeError:
            raw_data = h5f.create_group(self.results, 'raw_data')

        try:
            self.results.computed
        except tables.NoSuchNodeError:
            raw_data = h5f.create_group(self.results, 'computed')

        self.h5f = h5f
        # self.handler = DataHandler(h5f, self.log_info, self.log_err)

    def add_computed(self, paramspace_pt, all_data, overwrite=False):
        '''
        Adds data contained in all_data (as described in the main class doc string.
        
        If paramspace_pt is None, will add it directly in results.computed
        in groups defined by the all_data dictionary.
        If paramspace_pt is not None, the groups defined by all_data
        will be added in results.computed.<paramspace_pt_group>
        
        overwrite determines how to handle existing groups.
        True - Any data with the same path in both all_data and the file
                will be deleted from the file before the new data is stored.
        False - If any group defined by all_data exists, then the new data
                will be stored in those groups. If any data (not groups) of 
                the same name already exists, PyTables will throw an exception.
        '''
        if paramspace_pt is None:
            group = self._computed
        else:
            path = self.get_results_group_path(paramspace_pt)
            group = self.handler._nested_get_or_create_groups(self._computed, path)
        self.handler.store_data(group, all_data, overwrite)

    def add_raw_data(self, paramspace_pt, all_data):
        '''
        Same behaviour as add_computed, except that paramspace_pt is
        not optional and overwrite is not available (since raw data should
        not be overwritable). 
        See comment above for more details.
        '''
        path = self.get_results_group_path(paramspace_pt)
        group = self.handler._nested_get_or_create_groups(self._raw_data, path)
        self.handler.store_data(group, all_data, False)

    def add_sim_state(self, paramspace_pt, all_data):
        '''
        Same behaviour as add_computed, except that paramspace_pt is
        not optional and overwrite is not available (since sim_state should
        not be overwritable). 
        '''
        path = self.get_results_group_path(paramspace_pt)
        group = self.handler._nested_get_or_create_groups(self._sim_state, path)
        self.handler.store_data(group, all_data, False)
            
    def remove_computed_results(self, paramspace_pt, computation_name):
        results_group_path = self.get_results_group_path(paramspace_pt)
        ident = '{0}, {1}'.format(results_group_path, computation_name)
        self.log_info('-> Removing computed results for ' + ident, self.h5f)
        try:
            pspgroup = self._computed._f_get_child(results_group_path)
            compgroup = pspgroup._f_get_child(computation_name)
            compgroup._f_remove(recursive=True)
            self.h5f.flush()
        except tables.exceptions.NoSuchNodeError:
            print('Computed result does not exist ['+results_group_path+']: '+computation_name)
        self.log_info('<- Removed computed results for '+ident, self.h5f)

    # @staticmethod
    # def _maps_int_to_ndarray(data):
    #     """
    #     Checks whether data should be stored as a VLArray.
    #     :param data: dict
    #             If this is a mapping from integers to ndarrays, then it will be stored as a VLArray
    #     :return:
    #     """
    #     return all([isinstance(k, (int, np.integer)) and isinstance(v, np.ndarray)
    #                 for k, v in iteritems(data)])
    #
    # @staticmethod
    # def store_data(h5f, filters, group, all_data, overwrite=False):
    #     # If overwrite is enabled, we want to provide a list
    #     # of keys that should be deleted. This means any key
    #     # that maps to a non-dictionary (e.g. an array), or a
    #     # dictionary that stores a VLArray.
    #     if overwrite:
    #         todelete = [k for k, v in iteritems(all_data)
    #                     if not isinstance(v, dict) or ResultsTables._maps_int_to_ndarray(v)]
    #         # ident = '/'.join((parent._v_name, name))
    #         for node in group._f_iter_nodes():
    #             if node._v_name in todelete:
    #                 self.log_info(h5f, '!!! OVERWRITING ' + node._v_name)
    #                 node._f_remove(recursive=True)
    #
    #     for name, value in iteritems(all_data):
    #         if isinstance(value, dict):
    #             if ResultsTables._maps_int_to_ndarray(value):
    #                 ResultsTables._create_vlarray(h5f, group, name, value, filters)
    #             else:
    #                 subgroup = ResultsTables._single_get_or_create_group(h5f, group, name)
    #                 ResultsTables.store_data(h5f, filters, subgroup, value, overwrite)
    #         elif sparse.issparse(value):
    #             ResultsTables._store_sparse(h5f, group, name, value)
    #         elif isinstance(value, np.ndarray):
    #             ResultsTables._create_carray(h5f, group, name, value, filters)
    #         elif isinstance(value, mmap_array):
    #             try:
    #                 mmap = np.memmap(value.filename, value.dtype, 'r', shape=value.shape)
    #                 if value.T:
    #                     mmap = mmap.T
    #                 ResultsTables._create_carray(h5f, group, name, mmap, filters)
    #             except FileNotFoundError as e:
    #                 self.log_err(h5f, e)
    #         elif isinstance(value, str):
    #             h5f.create_array(group, name, value.encode())
    #         else:
    #             self.log_info(h5f, 'UNKNOWN TYPE IN DATA {} {}'.format(name, type(value)))
    #
    # @staticmethod
    # def _single_get_or_create_group(h5f, parent, name):
    #     '''
    #     It's necessary to have both this function and the below because if
    #     we combine them, the todelete list would not work correctly since
    #     names would have to be unique across all layers of the hierarchy.
    #     '''
    #     try:
    #         group = parent._f_get_child(name)
    #     except tables.NoSuchNodeError:
    #         group = h5f.create_group(parent, name)
    #     # else:
    #     #     if overwrite:
    #     #         ident = '/'.join((parent._v_name, name))
    #     #         self.log_info('!!! OVERWRITING group '+ident)
    #     #         for node in group._f_iter_nodes():
    #     #             if node._v_name in todelete:
    #     #                 node._f_remove(recursive=True)
    #     return group
    #
    # @staticmethod
    # def _nested_get_or_create_groups(h5f, parent, path):
    #     for name in path.split('/'):
    #         try:
    #             parent = parent._f_get_child(name)
    #         except tables.NoSuchNodeError:
    #             parent = h5f.create_group(parent, name)
    #     return parent
    #
    # @staticmethod
    # def _store_sparse(h5f, group, name, arr):
    #     if not sparse.isspmatrix_csr(arr):
    #         arr = arr.tocsr()
    #
    #     csr_group = h5f.create_group(group, name)
    #     csr_group._v_attrs.issparse = True
    #     if arr is not None and arr.nnz > 0:
    #         indptr, indices = csr_make_ints(arr.indptr, arr.indices)
    #         h5f.create_array(csr_group, 'data',   arr.data)
    #         h5f.create_array(csr_group, 'indptr', indptr)
    #         h5f.create_array(csr_group, 'indices',indices)
    #         h5f.create_array(csr_group, 'shape',  arr.shape)
    #     h5f.flush()
    #
    # @staticmethod
    # def _create_carray(h5f, group, name, data, filters):
    #     atom = tables.Atom.from_dtype(data.dtype)
    #     try:
    #         _d = h5f.create_carray(group, name, atom, data.shape, filters=filters)
    #         _d[:] = data
    #         h5f.flush()
    #     except Exception as e:
    #         self.log_err(h5f, 'EXCEPTION: {} {} {}'.format(name, np.ndim(data), e))
    #
    # @staticmethod
    # def _create_vlarray(h5f, group, name, data, filters):
    #     for v in data.values():
    #         dtype = v.dtype
    #     atom = tables.Atom.from_dtype(dtype)
    #     _d = h5f.create_vlarray(group, name, atom, filters=filters)
    #     for i in data.keys():
    #         _d.append(data[i])
    #     h5f.flush()

    # def add_spiketimes(self, paramspace_pt, source_name, spiketimes):
    #     resultsgroup_str = self.get_results_group_path(paramspace_pt)
    #     psp_spks_str = '/'.join((resultsgroup_str,'spikes'))
    #     group = self._nested_get_or_createGroups(self._raw_data, psp_spks_str)
    #
    #     ident = '{0}, {1}'.format(resultsgroup_str, source_name)
    #     all_data = {source_name:spiketimes}
    #     self._store_data(group, all_data, False)
    #     self.log_info('Added spike times for ' +ident)
    #
    # def add_population_rates(self, paramspace_pt, source_name, times, rates):
    #     resultsgroup_str = self.get_results_group_path(paramspace_pt)
    #     psp_rates_str = '/'.join((resultsgroup_str,'population_rates'))
    #     group = self._nested_get_or_createGroups(self._raw_data, psp_rates_str)
    #
    #     ident = '{0}, {1}'.format(resultsgroup_str, source_name)
    #     all_data = {source_name: {'times':times, 'rates':rates}}
    #     self._store_data(group, all_data, False)
    #     self.log_info('Added population rates for '+ident)
    #
    # def add_state_variables(self, paramspace_pt, source_name, variable_name,
    #                         times, values):
    #     resultsgroup_str = self.get_results_group_path(paramspace_pt)
    #     psp_statevar_str = '/'.join((resultsgroup_str,'state_variables'))
    #     group = self._nested_get_or_createGroups(self._raw_data, psp_statevar_str)
    #
    #     ident = '{0}, {1}, {2}'.format(resultsgroup_str, source_name, variable_name)
    #
    #     all_data = {source_name:{variable_name:{'times':times,'values':values}}}
    #     self._store_data(group, all_data, False)
    #     self.log_info('Added state variables for ' +ident)

    def add_log_file(self, paramspace_pt, log_file):
        '''
        Stores filetext in an Array called filename. The filetext parameter
        can be anything storeable as an Array, including a list of strings.
        '''
        resultsgroup_str = self.get_results_group_path(paramspace_pt)
        resultsgroup = self.handler._nested_get_or_create_groups(self._logs, resultsgroup_str)
        for name, text in log_file.items():
            # ident = '{0}, {1}'.format(resultsgroup_str, name)
            if isinstance(text, (list, tuple)):
                text = [t.encode() for t in text]
            elif isinstance(text, str):
                text = text.encode()
            if len(text):
                self.h5f.create_array(resultsgroup, name, text)
        self.h5f.flush()
        # self.log_info('Added log file for '+ident)
