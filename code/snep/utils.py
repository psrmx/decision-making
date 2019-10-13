from snep.library.brianimports import *
import numpy as np
from functools import reduce
from scipy.sparse import csr_matrix, lil_matrix, issparse
from six import iteritems
import sys
import os
import matplotlib.pyplot as plt
from collections import namedtuple
from typing import Tuple, Type, List, Union

'''
The wild import from snep.library.brian is there to bring in all the brian.units
and brian.stdunits, plus other brian class/function definitions, when available.
The rand, randn imports are necessary for similar reasons.
Those definitions are needed in the global namespace, throughout this file.
In other words: don't modify or remove them.
'''


def p2q(param):
    """'Parameter-to-Brian.Quantity converter"""
    if issparse(param.value):
        # Unfortunately it seems the Brian Units class doesn't know
        # how to multiply with sparse matrices.
        assert(isinstance(param.value,csr_matrix))
        data = unit_eval(param.value.data,param.units)
        quantity = csr_matrix((data, param.value.indices, param.value.indptr), param.value.shape)
        return quantity #csr_matrix(dense) if isinstance(param.value,csr_matrix) else lil_matrix(dense)
    elif isinstance(param.value, np.ndarray) and param.value.dtype.type is np.string_:
        return param.value
    else:
        return unit_eval(param.value,param.units)


class Parameter(object):
    """
    This is how we represent quantities in most of the snep framework. There are limitations with
    Brian Quantity objects which make them hard to work with when trying to load and store them into
    the hdf5 files. Consequently we keep our parameters (and anything with units) in this class
    as long as possible, only converting them to Brian Quantity objects when it makes sense.
    """
    quantity = property(fget=lambda self: p2q(self))
    def __init__(self, value, units='1', name=None):
        self.value = value
        self.units = units
        self.name = name
        
    def __repr__(self):
        units = '' if self.units == '1' else ','+self.units
        if self.name:
            name = ',name='+self.name
            value = str(type(self.value))
        else:
            name = ''
            value = self.value
        ret = 'Parameter({value}{units}{name})'.format(**locals())
        return ret
    
    def coordvalue(self):
        value = str(self) if self.name else self.value
        return value

    def __str__(self):
        if self.name:
            ret = self.name
        elif isinstance(self.value,str):
            ret = self.value
        else:
            value_str = str(self.value)
            ret = value_str if self.units == '1' else ' '.join((value_str,self.units))
        return ret

    def __cmp__(self, other):
        return not self.__eq__(other)

    def __eq__(self,other):
        return type(self) is type(other) and self.__dict__ == other.__dict__
    
    def __ne__(self,other):
        return not self.__eq__(other)


class ParameterArray(Parameter):
    """
    Same as Parameter but handles arrays of values and can be iterated on, returning single
    Parameter objects with the appropriate units.
    """
    def __init__(self, value, units='1', name=None):
        if not (isinstance(value,np.ndarray) or issparse(value) or isinstance(value,str)):
            value = np.array(value)
        Parameter.__init__(self, value, units, name)

    def __iter__(self):
#         return self
#     
#     def next(self):
        for p in self.value:
            yield Parameter(p, self.units, self.name)
        #raise StopIteration

    def __str__(self):
        if self.name:
            valuestr = self.name
        else:
            valuestr = str(self.value.shape)+str(self.value)
            if self.units != '1':
                valuestr = '->'.join((self.units,valuestr))
        return valuestr

    def __eq__(self,other):
        ret = False
        if type(self) is type(other):
            names = self.name==other.name
            units = self.units==other.units
            sparse = issparse(self.value) and issparse(other.value)
            dense = not issparse(self.value) and not issparse(other.value)
            if sparse:
                values = np.abs(self.value-other.value).nnz == 0 
            elif dense:
                if self.value.shape==other.value.shape and sum(self.value.shape) > 1:
                    values = (self.value==other.value).all()
                else:
                    values = self.value==other.value
            else:
                values = False
            
            ret = names and units and values
        return ret


class ParametersNamed(object):
    """
    Similar to ParameterArray, except that it can contain a list of homogeneous
    objects, each of which has a name. This allows us to have parameter ranges
    that are sets of arrays (which can be a mixture of sparse and dense arrays).

    Maybe this should inherit from dict, but it's not clear if that will
    interact with the existing code nicely.
    """
    def __init__(self, names_values, units='1'):
        self.units = units
        new_list = []
        for k, v in names_values:
            if isinstance(v, ParametersNamed):
                raise Exception('ParametersNamed only stores arrays!')
            elif isinstance(v, (Parameter,ParameterArray)):
                assert(units == v.units)
                v = v.value#self.names_values[k] = v.value
            new_list.append((k, v))
        self.names_values = new_list

    def get_quantity(self):
        return [(n, p2q(Parameter(v, self.units))) for n, v in self.names_values]
    quantity = property(fget=get_quantity)

#     def next(self):
#         for name,value in self.names_values:
#             return Parameter(value,self.units,name)
#         raise StopIteration

    def __iter__(self):
        for name, value in self.names_values:
            yield Parameter(value, self.units, name)
    
    def iteritems(self):
        for name, value in self.names_values:
            yield name, Parameter(value,self.units,name)#yield name, Parameter(value,self.units)

    def __str__(self):
        valuestr = ', '.join(x[0] for x in self.names_values)
        if self.units != '1':
            valuestr = '->'.join((self.units,valuestr))
        return valuestr
    
    def __eq__(self,other):
        return type(self) is type(other) and all(a == b for a, b in zip(self, other))

    def __ne__(self,other):
        return not self.__eq__(other)


def flatten_dict_of_dicts_the_one_line_version(params):
    """ Uses a Z-combinator to allow for recursion on an anonymous function"""
    Z = lambda f: (lambda x: f(lambda *args: x(x)(*args)))(lambda x: f(lambda *args: x(x)(*args)))
    return dict(Z(lambda f: lambda p,x: reduce(lambda l,k_v: l+f(p+(k_v[0],),k_v[1]), iteritems(x), [])\
                        if isinstance(x,dict) else [(p,x)])((),params))


def flatten_dict_of_dicts(p):
    """Turns dict of dictionaries into flat dict mapping path tuples to leaf values"""
    def rec(q, x):
        return reduce(lambda l, k_v: l+rec(q+(k_v[0],), k_v[1]), iteritems(x), []) if isinstance(x, dict) else [(q, x)]
    return dict(rec((), p))


def flatten(x):
    """Turns dict of dictionaries into flat list of leaves"""
    return flatten_dict_of_dicts(x).values()


def flatten_params_to_point(params):
    """Turns nested param dictionary into a flat dictionary mapping paths to values"""
    return flatten_dict_of_dicts(params)


def update_params_at_point(params, paramspace_pt, brian=False):
    """
    Given params, which is potentially nested dictionaries of parameters,
    and paramspace_pt which is a dictionary of path-tuples to parameters, we
    need to overwrite the appropriate values in the params dictionary.
    params = {'foo':0.0, 'bar':{'baz':'abc'}}
    paramspace_pt = {('foo'):2.0,  ('bar','baz'):'xyz'}
    """
    for name, value in iteritems(paramspace_pt):
        p = params
        for n in name[:-1]:
            if n not in p:
                p[n] = {}
            p = p[n]
        p[name[-1]] = value.quantity if brian and hasattr(value, 'quantity') else value


class ParameterSpace(object):
    """
    This class takes a dictionary of parameter ranges which specify a parameter space
    and produces a list of all the points in the space. It also allows two parameters to
    be linked, so that no product is taken over those two parameters (the restrictlist).

    It's basically a wrapper around a set of functions written by Konstantin.
    For a description of any function other than _transitive_set_reduce, talk to him since
    he wrote those.
    """
    def __init__(self, thedict, restrictlist):
        self.thedict = thedict
        self.restrictlist = self.transitive_set_reduce(restrictlist)

    @staticmethod
    def transitive_set_reduce(all_sets):
        """
        This function sanitises the user defined list of linked
        variables so that _preparedict doesn't fail.
        If any two sets have a common element, then those sets are added due to the transitive
        property of "linked variables".
        """
        if not all_sets:
            return []
        head, tail = set(all_sets[0]), [frozenset(t) for t in ParameterSpace.transitive_set_reduce(all_sets[1:])]
        disjoint = [t for t in tail if not head.intersection(t)] # all sets not intersecting with head
        head.update(*set(tail).difference(disjoint)) # add all other sets to head
        return [tuple(s) for s in [head]+disjoint]

    def _product(self, *args, **kwds):
        # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
        # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
        pools = map(tuple, args) 
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)
    
    def _preparedict(self, thedict,restrictlist):
        """
        Generates a dictionary with combined (keys as tuples, items as lists of lists)
        """
        otherdict = thedict.copy()
        dummydict = {}
        for elem in restrictlist:
            thelist = []
            for subelem in elem:
                thelist.append(otherdict.pop(subelem))
            dummydict[elem] = self.transposed(thelist)
        otherdict.update(dummydict)
        return otherdict
    
    def transposed(self, lists):
        if not lists: return []
        return map(lambda *row: list(row), *lists)

    def __iter__(self):
        a = self._preparedict(self.thedict,self.restrictlist)
        for valuesvec in self._product(*a.values()):
            b = {keys[0]:keys[1] for keys in zip([thekey for thekey in a.keys()],valuesvec)}
            c = {}
            for kkey in b:
                if isinstance(b[kkey],list):
                    for i,elem in enumerate(b[kkey]):
                        c[kkey[i]] = elem
                else:
                    c[kkey] = b[kkey]
            yield c


def set_brian_var(group, varname, value, localnamespace):
    """
    This is used for two things:
    1) Setting state variables on a NeuronGroup, where if the user specified a string, we
    need to evaluate it with a namespace defined by the neuron parameters before passing the
    resulting array to the group. NeuronGroups don't accept executable string expressions.
    2) Setting state variables on Synapses, which will accept executable string expressions, but
    we need to have the wildcard brian.units and brian.stdunits defined in the global namespace.
    """
    import re
    if isinstance(value, tuple):
        value, dt = value
        set_group_var_by_array(group, varname, TimedArray(value, dt=dt))
    else:
        # if isinstance(group,NeuronGroup) and isinstance(value,str):
        #     localnamespace['n_neurons'] = len(group)
        #     value = re.sub(r'\b' + 'rand\(\)', 'rand(n_neurons)', value)
        #     value = re.sub(r'\b' + 'randn\(\)', 'randn(n_neurons)', value)
        #     value = eval(value, globals(), localnamespace)
        group.__setattr__(varname, value)


def brian_monitors_to_rawdata(monitors):
    """
    Extracts the internal data from the Brian Monitors so that we don't have to pass Brian
    objects back out of the run function.
    :param monitors: dictionary of monitors from simulation
    :return: dictionary with data extracted from monitors
    """
    import time
    import traceback
    from six import iteritems
    rawdata = {'spikes': {}, 'population_rates': {}, 'state_variables': {}}

    start = time.time()

    for pop_name, mon in iteritems(monitors['spikes']):
        try:
            # ii,tt = np.array(mon.i), np.array(mon.t)
            # n = np.max(ii)
            # spks = {i:[] for i in xrange(n+1)}
            # for i,t in zip(ii,tt):
            #     spks[i].append(t)
            tmp = {i: np.asarray(t) for i, t in iteritems(mon.spike_trains())}
            for s in tmp.values():
                s.sort()
            rawdata['spikes'][pop_name] = tmp
        except:
            traceback.print_exc()
    for pop_name, mon in iteritems(monitors['poprate']):
        rawdata['population_rates'][pop_name] = {'times': np.array(mon.t), 'rates': np.array(mon.rate)}
    for pop_name, var_mons in iteritems(monitors['statevar']):
        for var_name, var_mon in iteritems(var_mons):
            if pop_name not in rawdata['state_variables']:
                rawdata['state_variables'][pop_name] = {}
            values = getattr(var_mon, var_name)
            rawdata['state_variables'][pop_name][var_name] = {'times': np.array(var_mon.t),
                                                              'values': np.array(values)}

    total = time.time() - start
    print("brian_monitors_to_rawdata: {0}".format(total))
    return rawdata


def brian_monitors_active(monitors, active):
    """
    :param monitors: Standard dictionary of monitor objects as generated by the @defaultmonitors decorator
    :param active: Boolean whether monitors are to be active or not
    :return: None
    """
    from six import itervalues
    for m in itervalues(monitors):
        for n in itervalues(m):
            if isinstance(n, dict):
                for o in itervalues(n):
                    o.active = active
            else:
                n.active = active


def clean_brian_state(state):
    state = {k: v for k, v in iteritems(state) if not ('ratemonitor' in k or 'spikemonitor' in k or 'statemonitor' in k)}
    def r(s):
        return s if isinstance(s, np.ndarray) else {k:r(v) for k,v in iteritems(s) if isinstance(v, dict) or (np.ndim(v) > 0 and v.shape[0] > 0)}
    return r(state)


def user_select_experiment_dir(get_task_dir, f_ext='h5'):
    import os
    from operator import itemgetter
    all_exps = []
    for subdir in os.listdir(get_task_dir):
        sdp = os.path.join(get_task_dir, subdir)
        if os.path.isdir(sdp):
            full_file_path = os.path.join(sdp, f'experiment.{f_ext}')
            if os.path.exists(full_file_path):
                all_exps.append((full_file_path, subdir))
                
    all_exps = sorted(all_exps, key=itemgetter(1))
    items = [subdir for ffp, subdir in all_exps]
    inp = user_select_from_list(items, 'Select an experiment')
    path, subdir = all_exps[inp]
    return path, subdir


def user_select_from_list(items, prompt):
    import six.moves
    if len(items) > 1:
        for i, item in enumerate(items):
            print('{0}: {1}'.format(i, item))
        inp = -2
        while inp not in range(-1, len(items)):
            try:
                inp = int(six.moves.input('{0} (0-{1}): '.format(prompt, len(items)-1)))
            except ValueError:
                print('Invalid selection')
    else:
        inp = 0
    return inp


def make_tables_from_path(path):
    """
    If given path is a file, then that is opened as the ExperimentTables source.
    Otherwise, if it is a directory, then we find all sub-directories that contain
    experiment.h5 files and ask the user to select one.
    """
    import os
    from snep.tables.experiment import ExperimentTables
    path = os.path.expanduser(path)
    if os.path.isdir(path):
        if 'experiment.h5' in os.listdir(path):
            path = os.path.join(path, 'experiment.h5')
        else:
            path, _subdir = user_select_experiment_dir(path)
    print('Opening experiment: ' + path)
    return ExperimentTables(path)


def filter_network_objs_for_run(allnetworkobjects):
    """
    Given a dictionary of all objects to be passed to the brian.Network
    constructor, we remove anything that is not a NeuronGroup, Connection
    or NetworkOperation. Any NeuronGroup subgroups are also removed since
    their parents are added.
    """
    canrun = (Connection,NetworkOperation)
    ano = flatten(allnetworkobjects)
    nosubgroups = [obj for obj in ano if obj is not None 
                                          and (isinstance(obj,canrun)
                                               or (isinstance(obj,NeuronGroup)
                                                   and hasattr(obj, '_owner')
                                                   and obj._owner == obj))]
    return nosubgroups


def make_square_figure(nsubplots):
    """
    Given a desired number of subplots, this returns the number
    of rows and columns that yields the closest thing to a square plot
    """
    """
    ncols = np.sqrt(nsubplots)
    int_cols = int(ncols)
    if ncols-int_cols < 1e-4:
        nrows = int_cols
    else:
        nrows = int(nsubplots / int_cols)
    ncols = int_cols
    nrows = nrows+1 if ncols * nrows < nsubplots else nrows
    assert(ncols * nrows >= nsubplots)
    return nrows, ncols"""
    ncols = int(np.ceil(np.sqrt(nsubplots)))
    while nsubplots % ncols != 0 and ncols > 0:
        ncols -= 1
    if ncols <= 0:
        raise Exception("Couldn't make a square figure")
    nrows = nsubplots // ncols
    return nrows, ncols


def csr_make_ints(indptr,indices):
    # This function ensures the index arrays are integer types
    # because numpy doesn't like it when they're floats.
    indptr = indptr  if indptr.dtype == np.int32 or indptr.dtype == np.int64 \
                        else indptr.astype(np.int32)
    indices= indices if indices.dtype == np.int32 or indices.dtype == np.int64 \
                        else indices.astype(np.int32)
    return indptr, indices


def write_named_param(h5f, aliased_group, name, value):
    sparse = isinstance(value, (lil_matrix,csr_matrix))
    if sparse:
        write_sparse(h5f, aliased_group, name, value)
    else:
        h5f.create_array(aliased_group, name, value)
    return sparse


def write_sparse(h5f, group, arrayname, value):
    lil = isinstance(value, lil_matrix)

    csr = value.tocsr() if lil else value
    csrgroup = h5f.create_group(group, arrayname)

    indptr, indices = csr_make_ints(csr.indptr, csr.indices)
    h5f.create_array(csrgroup, 'data', csr.data)
    h5f.create_array(csrgroup, 'indptr', indptr)
    h5f.create_array(csrgroup, 'indices', indices)
    h5f.create_array(csrgroup, 'shape', csr.shape)


def read_sparse(group, arrayname):
    csrgroup = group._f_get_child(arrayname)
    data   = csrgroup.data.read()
    indices= csrgroup.indices.read()
    indptr = csrgroup.indptr.read()
    shape  = csrgroup.shape.read()
    # The next few lines make sure the index arrays are integer types
    # because numpy doesn't like it when they're floats.
    indptr = indptr if indptr.dtype == np.int32 \
                            or indptr.dtype == np.int64 \
                        else indptr.astype(np.int32)
    indices= indices if indices.dtype == np.int32 \
                            or indices.dtype == np.int64 \
                        else indices.astype(np.int32)
    csr = csr_matrix((data,indices,indptr),shape=shape)
    return csr


class CompareExperiments(object):
    def __init__(self, experiment_A, experiment_B):
        self._a = experiment_A
        self._b = experiment_B
    
    def checkparams(self):
        pa = self._a.get_general_params(False)
        pb = self._b.get_general_params(False)
        missing_a, missing_b, modified = {},{},{},
        addpath = lambda p,k: p+(k,)
        CompareExperiments._recurse(pa, pb, missing_a, missing_b, modified, (), addpath)
        return {'missing_a':missing_a, 'missing_b':missing_b, 'modified':modified}
    
    def checkranges(self):
        ra = self._a.read_param_ranges()
        rb = self._b.read_param_ranges()
        missing_a, missing_b, modified = {},{},{},
        addpath = lambda p,k: k
        CompareExperiments._recurse(ra, rb, missing_a, missing_b, modified, None, addpath)
        return {'missing_a':missing_a, 'missing_b':missing_b, 'modified':modified}
    
    @staticmethod
    def _recurse(pa, pb, missing_a, missing_b, modified, path, addpath):
        sa,sb = set(pa),set(pb)
        missing_b.update({addpath(path,k):pa[k] for k in sa.difference(sb)})
        missing_a.update({addpath(path,k):pb[k] for k in sb.difference(sa)})
        for k in sa.intersection(sb):
            va,vb = pa[k], pb[k]
            ta,tb = type(va),type(vb)
            if ta is tb and ta is dict:
                CompareExperiments._recurse(va, vb, missing_a, missing_b, modified, addpath(path,k), addpath)
            elif ta is not tb or va != vb:
                modified[addpath(path,k)] = (va, vb)


def compare_experiments(path):
    exps = [make_tables_from_path(path) for _ in range(2)]
    for e in exps: e.open_file(True)
    ce = CompareExperiments(*exps)
    fixed = ce.checkparams()
    ranges = ce.checkranges()
    for e in exps: e.close_file()
    return {'fixed':fixed, 'ranges':ranges}


def plot_connections(connectivity, density=None, norm=1.):
    if density is not None:
        plt.figure()
        plt.imshow(density, interpolation='nearest')
    cdense = connectivity.toarray()
    c = np.zeros(cdense.shape[1])
    step = -float(cdense.shape[1])/cdense.shape[0]
    for i in range(cdense.shape[0]):
        r = cdense[i,:]
        r = np.roll(r,int(i*step))
        c += r
    c /= (cdense.shape[0] * norm)
    plt.figure()
    plt.plot(c)#np.log(c))
    plt.figure()
    plt.imshow(cdense, interpolation='nearest')
    return c


def compute_angle(a, b, in_radians=False):
    if not isinstance(a, np.ndarray):
        a = a.toarray()
    if not isinstance(b, np.ndarray):
        b = b.toarray()
    a_flat = a.flatten()
    b_flat = b.flatten()
    a_norm = np.linalg.norm(a_flat)
    b_norm = np.linalg.norm(b_flat)
    angle = np.arccos(np.dot(a_flat, b_flat) / (a_norm * b_norm))

    return angle if in_radians else np.rad2deg(angle)


if sys.version_info >= (3, ):
    def decode(s):
        return s.decode()
else:
    def decode(s):
        return s


def experiment_opener(file_names, load_path, readonly=True,
                      show=False, save=False, dpi=None, onlyfinished=False, onlyunfinished=False):
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    def inner(func):
        def wrapped(*args, **kwargs):
            path0 = os.path.expanduser(load_path)
            tables_task_ids = {}
            for ref, filename in file_names.items():
                path1 = os.path.join(path0, filename)
                print(ref, filename)
                tables = make_tables_from_path(path1)
                tables.open_file(readonly=readonly)
                if not readonly:
                    tables.initialize()
                task_ids = tables.get_task_ids(onlyunfinished=onlyunfinished, onlyfinished=onlyfinished)
                tables_task_ids[ref] = (tables, task_ids)
            files = func(tables_task_ids, *args, **kwargs)
            print('Finished plotting')
            for tables, task_ids in tables_task_ids.values():
                tables.close_file()
            if isinstance(files, dict):
                files = list(files.items())
            if isinstance(files, list):
                for f in files:
                    if isinstance(f, tuple):
                        file_name, fig = f
                        if save:
                            print('Saving {}'.format(file_name))
                            plt.figure(fig.number)
                            fig = plt.gcf()
                            if file_name[-3:] == 'svg':
                                canvas = FigureCanvasSVG(fig)
                                canvas.print_svg(file_name)
                            else:
                                fig.canvas.print_png(file_name)
                                # plt.savefig(file_name, dpi=dpi)
            if show:
                plt.show()
            print('Finished showing')
            return files
        return wrapped
    return inner


def filter_tasks(task_ids, targets):
    from functools import partial
    from operator import eq

    if not isinstance(targets, list):
        targets = [targets]

    for t in targets:
        for coord, v in list(t.items()):
            if not callable(v):
                if isinstance(v, (np.float, np.float32, np.float64)):
                    t[coord] = partial(np.isclose, v, atol=1e-10)
                else:
                    t[coord] = partial(eq, v)

    x = [tid for t in targets
                for tid in task_ids
                    if all(fn(tid[coord].value) for coord, fn in t.items())]
    print('Filtered from {} to {} task ids, using {}'.format(len(task_ids), len(x), targets))
    return x


def allocate_aligned(shape: Union[int, Tuple[int, ...]],
                     alignment: int=64, dtype: Type[np.number]=np.float32) -> np.ndarray:
    size = np.prod(shape)
    dtype = np.dtype(dtype)
    nbytes = size * dtype.itemsize
    buf = np.zeros(nbytes + alignment, dtype=np.uint8)
    start_index = -buf.ctypes.data % alignment
    a = buf[start_index:start_index + nbytes].view(dtype)
    return a.reshape(shape)


def reallocate_aligned(a: np.ndarray, alignment: int=64) -> np.ndarray:
    needs_realignment = a.ctypes.data % alignment
    # if needs_realignment:
    #     print('Array of shape {}, dtype {} required reallocation'.format(a.shape, a.dtype))
    b = allocate_aligned(a.shape, alignment, a.dtype)
    b[...] = a
    a = b
    assert not a.ctypes.data % alignment
    return a


def set_im(ax, ax_title: str, x_n: int, y_n: int,
           x_names: List[str], y_names: List[str],
           x_label: str, y_label: str) -> None:
    ax.set_title(ax_title)
    ax.set_xticks(np.linspace(0, x_n - 1, x_n))
    ax.set_yticks(np.linspace(0, y_n - 1, y_n))
    ax.set_xlim(-.5, x_n - .5)
    ax.set_ylim(-.5, y_n - .5)
    ax.set_xticklabels(x_names, rotation=45., minor=False)
    ax.set_yticklabels(y_names, rotation=45., minor=False)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
