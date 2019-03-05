from .network_base import *
from six import iteritems


class NetworkTablesReader(NetworkTablesReaderBase):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    
    '''
    These internal properties are defined so that we can restructure the hdf5 file without
    modifying the code that accesses it anywhere but here.
    '''
    _neurongroups = property(fget = lambda self: self.network.neurongroups)
    _synapses = property(fget = lambda self: self.network.synapses)

    def __init__(self):
        pass

    def as_dictionary(self, paramspace_pt, brian=True):
        '''
        This is used by the snep.experiment framework to convert the network definition as
        it is stored in the hdf5 file into a dictionary of Python objects and, if brian=True
        also Brian.Quantity objects where appropriate. This dictionary is then passed to
        the subprocess to construct and simulate the network.
        
        IMPORTANT: the paramspace_pt value overrides any values also specified on the populations
        and connections. In other words: the network can be blindly constructed in the
        subprocess without any knowledge of the paramspace_pt. 
        '''

        params = self.get_general_params(brian)
        if brian:
            paramspace_pt = {k: v.quantity for k, v in iteritems(paramspace_pt)}

        # we only will write psp items that are not conns, pops, or subpops, because
        # otherwise we should assume that they're actually defined elsewhere
        # in the network.
        #x = ['synapses', 'neurongroups', 'subgroups']
        #tmp_psp = {k: v for k, v in iteritems(paramspace_pt) if k[0] not in x}
        update_params_at_point(params, paramspace_pt, brian)
        return params

    def get_general_params(self, brian):
        params = dict(ng=self.get_neurongroups(brian),
                      sg=self.get_subgroups(brian),
                      sy=self.get_synapses(brian),)

        general_params = NetworkTablesReaderBase.get_general_params(self, brian)
        params.update(general_params)
        return params

    def get_neurongroups(self, brian):
        ''' brian2.NeuronGroup
        N : int
        model : (str, `Equations`)
        method : (str, function), optional
        threshold : str, optional
        reset : str, optional
        refractory : {str, `Quantity`}, optional
        namespace: dict, optional
        name : str, optional
        '''
        cols =  getattr(NeuronGroup, 'columns').keys()
        neurongroups = self._get_groups(brian, cols, self._neurongroups, self._neurongroups.all)
        return neurongroups

    def get_subgroups(self, brian):
        cols =  getattr(Subgroup, 'columns').keys()
        neurongroups = self._get_groups(brian, cols, self._neurongroups, self._neurongroups.subgroups)
        return neurongroups

    def get_synapses(self, brian):
        ''' brian2.Synapses
        source : `SpikeSource`
        target : `Group`, optional
        model : {`str`, `Equations`}, optional
        pre : {str, dict}, optional
        post : {str, dict}, optional
        connect : {str, bool}. optional
        delay : {`Quantity`, dict}, optional
        namespace : dict, optional
        method : {str, `StateUpdateMethod`}, optional
        name : str, optional
        connect_prepost : array, optional
        '''
        cols =  getattr(Synapses, 'columns').keys()
        synapses = self._get_groups(brian, cols, self._synapses, self._synapses.all)
        return synapses

    def _get_groups(self, brian, cols, group, table):
        groups = {}
        for gr in table:
            gd = {}
            name = gr['name'] 
            gd['name'] = name
            for c in cols:
                gd[c] = gr[c] 
            gr = group._f_get_child(name)
            namespace = self._read_named_variant_table(gr, gr.namespace, brian)
            gd.update(namespace)
            svs = self._read_timed_variant_table(gr.state_variable_setters, brian)
            assert('svs' not in gd)
            gd['svs'] = svs
            groups[name] = gd

        return groups


class NetworkTables(NetworkTablesReader, NetworkTablesBase):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        NetworkTablesBase.initialize(self, h5f, parentgroup)

        try:
            self._neurongroups
        except tables.exceptions.NoSuchNodeError:
            self.h5f.create_group(self.network,'neurongroups')
            self.h5f.create_table(self._neurongroups, 'all', NeuronGroup, "All NeuronGroups")
            self.h5f.create_table(self._neurongroups, 'subgroups', Subgroup)
        try:
            self._synapses
        except tables.exceptions.NoSuchNodeError:
            self.h5f.create_group(self.network,'synapses')
            self.h5f.create_table(self._synapses, 'all', Synapses, "All Synapses")
        
    def set_simulation(self, runtime, dt, target, others={}):
        ''' 
        Set the simulation parameters such as the step
        size (dt) and total simulated time (runtime)
        '''
        params = { 'dt':dt,
                   'runtime':runtime,
                   'target':target}
        params.update(others)
        self.add_parameters(params)
    
    def add_neurongroup(self, name, N, model, namespace, threshold='', reset='', method='', refractory=''):
        ''' brian2.NeuronGroup
        N : int
        model : (str, `Equations`)
        method : (str, function), optional
        threshold : str, optional
        reset : str, optional
        refractory : {str, `Quantity`}, optional
        namespace: dict, optional
        name : str, optional
        '''
        ng = self._neurongroups.all.row
        ng['N'         ] = N
        ng['name'      ] = name
        ng['model'     ] = model 
        ng['method'    ] = method
        ng['threshold' ] = threshold 
        ng['reset'     ] = reset 
        ng['refractory'] = refractory
        ng.append()
        self._neurongroups.all.flush()

        assert set(getattr(NeuronGroup, 'columns').keys()).isdisjoint(namespace.keys()), 'Name collision in NeuronGroup' 

        self._add_groups_tables(name, self._neurongroups, namespace)
    
    def add_subgroup(self, name, parent, start, size, non_default_params={}):
        subpop = self._neurongroups.subgroups.row
        subpop['super'] = parent
        subpop['name'] = name
        subpop['start'] = start
        subpop['size'] = size
        subpop.append()
        self._neurongroups.subgroups.flush()
        self._add_groups_tables(name, self._neurongroups, non_default_params)

    def add_synapse(self, name, source, target, namespace, 
                    model='', pre='', post='', connect='', prepost=None, delay='', method=''):
        ''' brian2.Synapses
        source : `SpikeSource`
        target : `Group`, optional
        model : {`str`, `Equations`}, optional
        pre : {str, dict}, optional
        post : {str, dict}, optional
        connect : {str, bool}. optional
        delay : {`Quantity`, dict}, optional
        namespace : dict, optional
        method : {str, `StateUpdateMethod`}, optional
        name : str, optional
        prepost : array or dictionary, optional
        '''
        sy = self._synapses.all.row
        sy['name'   ] = name
        sy['source' ] = source
        sy['target' ] = target
        sy['model'  ] = model
        sy['pre'    ] = pre
        sy['post'   ] = post
        sy['connect'] = connect
        sy['method' ] = method
        sy.append()
        self._synapses.all.flush()
        
        if prepost is not None:
            assert connect=='', "You cannot specify 'connection' and 'prepost' simultaneously"
            assert 'prepost' not in namespace
            namespace = dict(namespace) # copy namespace before modifying
            namespace['prepost'] = prepost
        
        assert set(getattr(Synapses, 'columns').keys()).isdisjoint(namespace.keys()), 'Name collision in Synapses'
        
        self._add_groups_tables(name, self._synapses, namespace)

    def _add_groups_tables(self, name, group, namespace):
        sg = self.h5f.create_group(group, name)
        svsg = self.h5f.create_group(sg, 'state_variable_setters')
        self.h5f.create_table(svsg,'all', TimedVariantType)
        paramtable = self.h5f.create_table(sg, 'namespace', NamedVariantType, name+" default params")
        self._write_named_variant_table(sg, paramtable, namespace)
        
    def add_neurongroup_state_variable_setters(self, ngname, svs):
        svsgroup = self._neurongroups._f_get_child(ngname).state_variable_setters
        self._write_timed_variant_table(svsgroup, svs)

    def add_synapse_state_variable_setters(self, syname, svs):
        svsgroup = self._synapses._f_get_child(syname).state_variable_setters
        self._write_timed_variant_table(svsgroup, svs)

    def delete_all_parameter_ranges(self):
#         self._reset_parameter_range_groups(self._neurongroups)
#         self._reset_parameter_range_groups(self._populations)
#         self._reset_parameter_range_groups(self._synapses)
#         self._reset_parameter_range_groups(self._connections)
        self.h5f.remove_node(self.network, 'linked_parameter_ranges')
        self.h5f.create_table(self.network, 'linked_parameter_ranges',
                     LinkedRanges, "Linked parameter ranges")
        self.h5f.flush()

