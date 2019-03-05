import tables
from snep.utils import Parameter, ParameterArray, ParametersNamed, \
                        update_params_at_point, flatten_params_to_point, \
                        write_sparse, read_sparse, write_named_param, decode
from ..rows import NamedVariantType, TimedVariantType, AliasedParameters, \
                   LinkedRanges, Subgroup, NeuronGroup, Synapses
from six import iteritems
import numpy as np


class NetworkTablesReaderBase(object):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to read a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''

    _parameters = property(fget=lambda self: self.network.parameters)
    _linked = property(fget=lambda self: self.network.linked_parameter_ranges)
    _parameter_ranges = property(fget=lambda self: self.network.parameter_ranges)

    def set_root(self, network):
        '''
        network - a group in the ExperimentTables hdf5 file.
        '''
        self.network = network

    def get_general_params(self, brian):
        '''
        Returns a dictionary of dictionaries containing the parameters
        common to all simulations in this experiment. In other words,
        everything not defined by the parameter ranges.
        '''
        group = self._parameters
        table = self._parameters.all
        params = self._read_named_variant_table(group, table, brian)
        return params

    def _user_defined_links(self):
        '''
        Returns a list of all the user defined linked variables as tuples such 
        as (('exc', 'C'), ('inh', 'g_l')) which means that those parameters should be varied
        together over their ranges.
        '''
        all_links = []
        for l in self._linked:
            coord_a, coord_b = decode(l['coord_a']), decode(l['coord_b'])
            link = (self._get_tuple_for_coord(coord_a),  # defined in tables.paramspace
                    self._get_tuple_for_coord(coord_b))
            all_links.append(link)
        return all_links

    def _read_named_variant_table(self, group, table, brian):
        ''' Returns a dictionary that maps parameter names to parameter values from a given table '''
        return {decode(row['name']): self._get_variant_type(group, row, 'vartype', brian)
                for row in table}

    def _read_timed_variant_table(self, group, brian):
        all_tvs = {}
        for tv in group.all:
            tv_vals = self._get_variant_type(group, tv, 'namedvartype/vartype', brian)
            dt = Parameter(tv['dt/value'], decode(tv['dt/units']))
            if dt.units != '1':
                # If the clock step dt has units, this is a timed array, so we make
                # a tuple of the values and the clock step so that a TimedArray object
                # can be constructed in the subprocess (TimedArray cannot be properly pickled
                # so we can't construct here).
                tv_vals = (tv_vals, dt.quantity if brian else dt)
            all_tvs[decode(tv['namedvartype/name'])] = tv_vals
        return all_tvs

    @staticmethod
    def _read_parameters_named(group, units):
        y = lambda x: x if (not hasattr(x, 'ndim') or x.ndim) else x[()]
        table = group._f_get_child('aliasedparameters')
        allparams = [(decode(r['name']), read_sparse(group, decode(r['name']))
                     if r['issparse'] else y(group._f_get_child(decode(r['name'])).read()))
                     for r in table]
        variant = ParametersNamed(allparams, units)
        return variant

    def _get_variant_type(self, group, row, field, brian):
        '''
        This reads a special variant type of data from a table. The data can be a short
        string, a single Parameter or a ParameterArray. This permits explicit support for
        a Brian feature which allows the user to define certain network parameters in all
        of those ways (see how the Brian Synapse class defines delays for an example).
        '''
        import numpy as np
        vartype = row[field+'/vartype']
        if vartype == b'strcode':
            variant = decode(row[field+'/varstring'])
        elif vartype == b'table':
            subgroup = group._f_get_child(decode(row[field+'/varstring']))
            subtable = subgroup._f_get_child('all')
            variant = self._read_named_variant_table(subgroup, subtable, brian)
        elif vartype == b'named':
            subgroup = group._f_get_child(decode(row[field+'/varstring']))
            variant = self._read_parameters_named(subgroup, decode(row[field+'/units']))
        else:
            if vartype == b'array':
                a = group._f_get_child(decode(row[field+'/varstring'])).read()
                if isinstance(a, list):
                    a = np.array(a)
                if a.dtype.type == np.bytes_:
                    a = np.array(a, dtype=np.str_)
                variant = ParameterArray(a, decode(row[field + '/units']))
            elif vartype == b'sparse':
                csr = read_sparse(group, decode(row[field+'/varstring']))
#                data   = csrgroup.data.read()
#                indices= csrgroup.indices.read()
#                indptr = csrgroup.indptr.read()
#                shape = csrgroup.shape.read()
#                # The next few lines make sure the index arrays are integer types
#                # because numpy doesn't like it when they're floats.
#                indptr = indptr if indptr.dtype == np.int32 \
#                                        or indptr.dtype == np.int64 \
#                                    else indptr.astype(np.int32)
#                indices= indices if indices.dtype == np.int32 \
#                                        or indices.dtype == np.int64 \
#                                    else indices.astype(np.int32)
#                csr = csr_matrix((data,indices,indptr),shape=shape)
                variant = ParameterArray(csr, decode(row[field+'/units']))
            elif vartype == b'float':
                variant = Parameter(row[field+'/varflt'], decode(row[field+'/units']))
            elif vartype == b'integer':
                variant = Parameter(row[field+'/varint'], decode(row[field+'/units']))
            else:
                raise Exception('Unknown type was stored in variant table: {0}'.format(vartype))
            if brian:
                variant = variant.quantity
        return variant

    def read_param_ranges(self):
        '''
        This is called from paramspace.ParameterSpaceTables.build_parameter_space
        
        Returns a dictionary mapping from parameter identifiers (a tuple containing
        the parameter owner and parameter name) to a ParameterArray of values.
        '''
        group = self._parameter_ranges
        table = self._parameter_ranges.all
        param_ranges = self._read_named_variant_table(group, table, False)
        param_ranges = flatten_params_to_point(param_ranges)
        return param_ranges
    
    def _read_param_links(self):
        '''
        This is called from paramspace.ParameterSpaceTables.build_parameter_space
        
        Returns a list of linked variables
        '''
        param_links = self._user_defined_links()
        return param_links

    def copy_network(self, destination_tables):
        self.network._f_copy_children(destination_tables.network,
                                      recursive=True, overwrite=True)


class NetworkTablesBase(object):
    # __metaclass__ = LockAllFunctionsMeta
    '''
    A class that knows how to create a subtree of the hdf5 file defined by ExperimentTables. That
    subtree specifies everything about the network and simulation that is part of the experiment.
    '''
    def __init__(self):
        self.h5f = None
        self.seed_map = {}

    def initialize(self, h5f, parentgroup):
        '''
        Given a parent group this function creates all the necessary default groups and
        tables that this class is responsible for.
        '''
        self.h5f = h5f
        try:
            network = parentgroup.network
        except tables.exceptions.NoSuchNodeError:
            network = self.h5f.create_group(parentgroup, 'network')
        self.set_root(network)
        self.seed_map = {}

        try:
            self._parameters
        except tables.exceptions.NoSuchNodeError:
            parameters = self.h5f.create_group(self.network, 'parameters')
            self.h5f.create_table(parameters, 'all', NamedVariantType, "General Parameters")
        try:
            rangegroup = self._parameter_ranges
        except tables.exceptions.NoSuchNodeError:
            rangegroup = self.h5f.create_group(self.network, 'parameter_ranges')
            self.h5f.create_table(rangegroup, 'all', NamedVariantType, "General Parameter Ranges")
        try:
            self.network.linked_parameter_ranges
        except tables.exceptions.NoSuchNodeError:
            self.h5f.create_table(self.network, 'linked_parameter_ranges',
                                  LinkedRanges, "Linked parameter ranges")

    def add_parameters(self, params):
        '''
        Adds the provided dictionary of dictionaries to the global parameters
        table. For brian networks, this is currently only used for a few
        things like simulation time step, etc.
        '''
        paramsgroup = self._parameters
        paramstable = self._parameters.all
        self._write_named_variant_table(paramsgroup, paramstable, params)

    def add_parameter_ranges(self, param_ranges):
        '''
        Add a range of values for each parameter listed in the provided
        dictionary of dictionaries. The cartesian product of these ranges
        form the parameter space. If two ranges should not form a subspace,
        but rather co-vary, they can be linked using link_parameter_ranges.
        '''
        prgroup = self._parameter_ranges
        prtable = prgroup.all
        self._update_coord_map(flatten_params_to_point(param_ranges).keys())
        self._write_named_variant_table(prgroup, prtable, param_ranges)

    def add_parameter_range(self, param_id, param_values):
        '''
        Singular of add_parameter_ranges
        '''
        self.add_parameter_ranges({param_id:param_values})

    def _write_named_variant_table(self, group, table, variants):
        '''
        Any table of variant types needs to be defined in its own group, so that we can
        freely save ParameterArrays as array nodes without worrying about name collisions.
        The variant type table must be called 'all'.
        '''
        row = table.row
        for name, value in iteritems(variants):
            row['name'] = name
            self._set_variant_type(group, row, 'vartype', value, name)
            if isinstance(value, dict):
                subgroup = self.h5f.create_group(group, name)
                subtable = self.h5f.create_table(subgroup, 'all', NamedVariantType)
                self._write_named_variant_table(subgroup, subtable, value)
            row.append()
        table.flush()

    def _write_timed_variant_table(self, group, timedvars):
        '''
        Same as _write_named_variant_table but with TimeVariantType rows
        rather than NamedVariantType rows.
        The timedvars parameter is a dictionary mapping from a variable name to either
        values, strings, arrays, or a tuple containing an array and a Parameter with units
        of time. This last case corresponds to the clock dt for a time varying variable.
        '''
        row = group.all.row
        for varname, values in iteritems(timedvars):
            if isinstance(values,tuple):
                values, dt = values
            else:
                dt = Parameter(0)
            row['dt/value'] = dt.value
            row['dt/units'] = dt.units
            row['namedvartype/name'] = varname
            self._set_variant_type(group, row, 'namedvartype/vartype', values, varname)
            row.append()
        group.all.flush()

    def _set_variant_type(self, group, row, field, variant, name=None):
        '''
        See the description of _get_variant_type on NetworkTablesReader class for an explanation
        of variant types.
        '''
        from scipy.sparse import lil_matrix, csr_matrix
        import numpy as np
        if isinstance(variant, str):
            row[field+'/vartype'] = 'strcode'
            row[field+'/varstring'] = variant
        elif isinstance(variant, ParametersNamed):
            row[field+'/vartype'] = 'named'
            arrayname = 'named_' + name
            row[field+'/varstring'] = arrayname
            row[field+'/units'] = variant.units
            subgroup = self.h5f.create_group(group, arrayname)
            self._write_parameters_named(subgroup, variant)
        elif isinstance(variant, ParameterArray):
            lil = isinstance(variant.value, lil_matrix)
            if lil or isinstance(variant.value, csr_matrix):
                row[field+'/vartype'] = 'sparse'
                arrayname = 'sparse_' + name
                row[field+'/varstring'] = arrayname
                row[field+'/units'] = variant.units
                try:
                    group._f_get_child(arrayname)
                except:
                    write_sparse(self.h5f, group, arrayname, variant.value)
            else:
                row[field+'/vartype'] = 'array'
                arrayname = 'ndarray_' + name
                row[field+'/varstring'] = arrayname
                row[field+'/units'] = variant.units
                try:
                    group._f_get_child(arrayname)
                except tables.NoSuchNodeError:
                    a = variant.value
                    if a.dtype.type in (np.string_, np.str_):
                        a = np.array(a, dtype=np.bytes_)
                    self.h5f.create_array(group, arrayname, a)
        elif isinstance(variant, dict):
            row[field+'/vartype'] = 'table'
            row[field+'/varstring'] = name
        else:
            if isinstance(variant, Parameter):
                value = variant.value
                units = variant.units
            else:
                value = variant
                units = '1'
            if isinstance(value, (float, np.float32, np.float64)):
                row[field+'/vartype'] = 'float'
                row[field+'/varflt'] = value
            elif isinstance(value, (int, np.int32)):
                row[field+'/vartype'] = 'integer'
                row[field+'/varint'] = value
            else:
                raise Exception('Unhandled type passed as Parameter. {} {} {}'.format(group, variant, name))
            row[field+'/units'] = units

    def _write_parameters_named(self, group, variant):
        table = self.h5f.create_table(group, 'aliasedparameters', AliasedParameters)
        row = table.row
        for name, param in variant.iteritems():
            row['name'] = name
            try:
                child = group._f_get_child(name)
                issparse = isinstance(child, tables.Group)
            except:
                issparse = write_named_param(self.h5f, group, name, param.value)
            row['issparse'] = issparse
            row.append()
        table.flush()
        
    def link_parameter_ranges(self, linked_params, show_only_first=False):
        '''
        linked_params - a list of parameters to be linked to each other. Each
                    element should be a tuple containing strings that describe
                    the path to the linked parameter
        '''
        if show_only_first:
            for l in linked_params:
                assert l in self.coord_map, "Parameter ranges must be added before they can be linked. {}".format(l)
            self.coord_map.update({x: -1 for x in linked_params[1:]})
            self.coord_map.update({y: j for j, y in enumerate(x for x, i in iteritems(self.coord_map) if i >= 0)})
            
        r = self._linked.row
        for a, b in zip(linked_params[:-1], linked_params[1:]):
            r['coord_a'] = self.make_column_from_coord(a)
            r['coord_b'] = self.make_column_from_coord(b)
            r.append()
        self._linked.flush()

    def _define_seeds(self, param_ranges):
        #import random # no need to seed with time, since it's done automatically on import
        iters = param_ranges[('iter',)].value.size if ('iter',) in param_ranges else 1
        ii32 = np.iinfo(np.int32)
        if iters > 1:
            seeds = ParameterArray(np.random.randint(ii32.max, size=iters).astype(dtype=np.int32))
            self.add_parameter_ranges({'seed': seeds})
            self.link_parameter_ranges([('iter',), ('seed',)])
            param_ranges.update({('seed',): seeds})
        else:
            seed = Parameter(np.random.randint(ii32.max, size=1).astype(dtype=np.int32)[0])
            self.add_parameters({'seed': seed})

        return param_ranges

    def new_seeds(self):
        ii32 = np.iinfo(np.int32)
        gp = self.get_general_params(False)
        if 'seed' in gp:
            seed = np.random.randint(ii32.max, size=1).astype(dtype=np.int32)[0]
            for row in self._parameters.all:
                if row['name'] == b'seed':
                    row['vartype/varint'] = seed
            self._parameters.all.flush()
        else:
            pr = self.read_param_ranges()
            seeds = pr[('seed',)].value
            new_seeds = np.random.randint(ii32.max, size=seeds.size).astype(dtype=np.int32)
            seed_map = {old: new for old, new in zip(seeds, new_seeds)}
            arr = self._parameter_ranges._f_get_child('ndarray_seed')
            arr[:] = new_seeds
            arr.flush()
            self.seed_map = seed_map
            return seed_map
