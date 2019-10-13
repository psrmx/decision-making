from ...utils import set_brian_var, brian_monitors_to_rawdata
from ...decorators import defaultmonitors
from six import iteritems


@defaultmonitors
def preproc(task_info, taskdir, tempdir, simulation_objects, local_objects, monitors):
    '''
    Add additional monitors to the network, network operators or
    even additional neuron groups and connections.

    neuron_groups - dictionary of Brian NeuronGroup and Synapse objects.
    monitors - dictionary describing which dictionaries are to be built.

    Returns the original neuron_groups (including any new groups added) and
    a dictionary of Brian Monitor objects.

    The @defaultmonitors decorator will construct any standard Brian monitors:
    PopulationRateMonitor, SpikeMonitor, StateMonitor.
    '''
    monitor_objs = {}
    return simulation_objects, local_objects, monitor_objs


def run(task_info, taskdir, tempdir):
    '''
    The most basic run implementation possible: it simply constructors a Network, runs it
    for the requested time and then extracts the raw data from all monitors.
    '''
    from snep.utils import flatten
    from brian2 import reinit
    from brian2 import Network, device, second, Nameable
    from brian2.devices.device import CurrentDeviceProxy

    sim_objects, local_objects = make(task_info, taskdir)
    sim_objects, local_objects, monitor_objs = preproc(task_info, taskdir, tempdir, sim_objects,
                                                       local_objects, task_info['monitors'])

    locals().update(local_objects)

    network_objs = list(set(sim_objects.values() + flatten(monitor_objs)))
    net = Network(network_objs)

    rt = task_info['runtime']
    net.run(rt, report='text')

    target = task_info['target']
    if target == 'standalone':
        build = CurrentDeviceProxy.__getattr__(device, 'build')
        build(directory=tempdir, compile=True, run=True, debug=False)

    rawdata = brian_monitors_to_rawdata(monitor_objs)
    results = postproc(sim_objects, task_info, rawdata)
    return results


def postproc(simulation_objects, task_info, rawdata):
    '''
    This is where any computations on the rawdata should be done. Results
    that are to be stored should be returned in a dictionary. Any raw data you
    want saved to the HDF5 file should be returned in the same dictionary.

    By default we just return the rawdata, which means all recorded data
    will be saved.
    '''
    return rawdata


def make(params, taskdir):
    '''
    The default network constructor. Takes a set of dictionaries that specify the network
    and returns a dictionary of Brian NeuronGroups and Synapses.
    '''
    from brian2 import defaultclock, prefs, set_device, device
    from brian2.devices.device import CurrentDeviceProxy

    target = params['target']
    if target == 'standalone':
        set_device('cpp_standalone', build_on_run=False)
    else:
        prefs.codegen.target = target
        reinit = CurrentDeviceProxy.__getattr__(device, 'reinit')
        reinit()

    if target == 'standalone':
        insert_code = CurrentDeviceProxy.__getattr__(device, 'insert_code')
        insert_code('main', 'srand('+str(int(params['seed']))+');')
        #prefs.devices.cpp_standalone.openmp_threads = 4
        #os.environ['MKL_DYNAMIC'] = 'true'
        #os.environ['OMP_NESTED'] = 'true'

    defaultclock.dt = params['dt']

    neurongroups = params['ng']
    subgroups = params['sg']
    synapses = params['sy']
    resume_state = params['resume_state'] if 'resume_state' in params else {}
    resume_state = dict(resume_state)

    sim_objects = {}
    for ngname, ng_info in iteritems(neurongroups):
        rs = resume_state.pop(ngname, {})
        sim_objects[ngname] = make_neurongroup(dict(ng_info), dict(rs))

    for sgname, sg_info in iteritems(subgroups):
        rs = resume_state.pop(sgname, {})
        sim_objects[sgname] = make_subgroup(sim_objects, dict(sg_info), dict(rs))

    for syname, sy_info in iteritems(synapses):
        rs = resume_state.pop(syname, {})
        sim_objects[syname] = make_synapses(sim_objects, dict(sy_info), dict(rs))

    local_objects = {}
    return sim_objects, local_objects, resume_state


def make_neurongroup(ng_info, rs):
    from brian2 import NeuronGroup
    keys = ['N', 'name', 'model', 'method', 'threshold', 'reset', 'refractory']
    all_svs = ng_info.pop('svs')
    kwargs = {k: v for k, v in iteritems(ng_info) if k in keys and v != ''}
    kwargs['namespace'] = {k: v for k, v in iteritems(ng_info) if k not in keys}
    ng = NeuronGroup(**kwargs)
    ng.set_states(rs, units=False)
    all_svs = {k: v for k, v in iteritems(all_svs) if k not in rs}
    assert not len(ng_info), "set_brian_var not using namespace"
    for var, svs in iteritems(all_svs):
        set_brian_var(ng, var, svs, ng_info)
    return ng


def make_subgroup(sim_objects, sg_info, rs):
    from brian2 import Subgroup
    keys = ['name', 'size', 'start', 'super', 'svs']
    for k, v in iteritems(sg_info):
        if k not in keys:
            raise ValueError("Parameter values have to be set in the parent NeuronGroup")
    parent = sim_objects[sg_info['super']]
    name = sg_info['name']
    start = sg_info['start']
    size = sg_info['size']
    all_svs = sg_info['svs']
    sg = Subgroup(parent, start, start+size, name)
    sg.set_states(rs, units=False)
    all_svs = {k: v for k, v in iteritems(all_svs) if k not in rs}
    assert not len(sg_info), "set_brian_var not using namespace"
    for var, svs in iteritems(all_svs):
        set_brian_var(sg, var, svs, sg_info)
    return sg


def make_synapses(sim_objects, sy_info, rs):
    from brian2 import Synapses
    from snep.library.structure import make_connections_fixed_in
    import numpy as np
    keys = ['name', 'source', 'target', 'model', 'pre', 'post', 'connect', 'method', 'delay']
    all_svs = sy_info.pop('svs')
    prepost = sy_info.pop('prepost', None)
    rescale_pow = sy_info.pop('rescale_pow', None)

    source = sim_objects[sy_info['source']]
    target = sim_objects[sy_info['target']]
    kwargs = {k: v for k, v in iteritems(sy_info) if k in keys and v != ''}
    kwargs['source'] = source
    kwargs['target'] = target
    kwargs['namespace'] = {k: v for k, v in iteritems(sy_info) if k not in keys}

    kwargs['namespace'].setdefault('g_rescale', 1.)
    if 'p' in prepost and rescale_pow:
        c = prepost['p'] * len(source)
        kwargs['namespace']['g_rescale'] /= np.power(c, rescale_pow)
    if 'g' in kwargs['namespace']:
        kwargs['namespace']['g'] *= kwargs['namespace']['g_rescale']

    c_str = ' {} -> {} connection'.format(sy_info['source'], sy_info['target'])
    sy = Synapses(**kwargs)
    if set(['i', 'j']).issubset(rs):
        print('Restoring'+c_str)
        sy.connect(rs.pop('i'), rs.pop('j'))
    elif prepost is not None:
        if isinstance(prepost, dict):
            print('Computing'+c_str)
            cpre, cpost = make_connections_fixed_in(len(source), len(target), prepost)
        else:
            print('Setting'+c_str)
            cpre, cpost = prepost[:, 0], prepost[:, 1]
        sy.connect(cpre, cpost)
    sy.set_states(rs, units=False)
    all_svs = {k: v for k, v in iteritems(all_svs) if k not in rs}
    assert not len(sy_info), "set_brian_var not using namespace"
    for var, svs in iteritems(all_svs):
        set_brian_var(sy, var, svs, sy_info)
    return sy

# def load_state_vars_or_weights(pt_name, svs, path):
#     import cPickle
#     if isinstance(svs, str) and 'file:' in svs:
#         svs = svs.replace('file:','')
#         pn = find_similar_picklefile_in_dir(svs, pt_name)
#         with open(pn, 'rb') as f:
#             svs = cPickle.load(f)
#             for n in path:
#                 svs = svs[n]
#     return svs
#
# def find_similar_picklefile_in_dir(dirname, name):
#     found = False
#     for fn in os.listdir(dirname):
#         pn = os.path.join(dirname,fn)
#         if not os.path.isdir(pn) and ('.pickle' in fn) and (name in fn or fn.replace('.pickle','') in name):
#             found = True
#             break
#     assert(found)
#     return pn

# def make_poisson_input(ipopname, ipop_info, synapses, input_connections,
#                        neuron_groups, clock, solver):
#     from brian2 import PoissonInput, PoissonGroup
#
#     for ic_name,ic in iteritems(input_connections):
#         if ic['popname_pre'] == ipopname:
#             break
#     if ic['popname_pre'] != ipopname:
#         print("Unconnected Poisson input")
#         pi = None
#     else:
#         popparams = ipop_info['params']
#         rate = popparams['rate']
#         N = ipop_info['size']
#
#         syn = synapses[ic['synapse']]
#         target_name = ic['popname_post']
#         target = neuron_groups[target_name]
#
#         if ipop_info['model'] == 'PoissonGroup':
#             pi = PoissonGroup(N=N, rates=rate, clock=clock)
#             #conn = make_connection_old(pi, target, ic, syn, clock, solver)
#             #neuron_groups[ic_name] = conn
#         else:
#             state = syn['equations']['eqs_model']
#             weight = popparams['weight']
#             pi = PoissonInput(target=target, N=N, rate=rate, weight=weight, state=state)
#     return pi
#
# def make_poisson_group():
#     pass
#
#
# def make_connection_experimental(pt_name, pops, source, target, con_info, synapse, clock, solver):
#     '''
#     Similar to make_population, this function makes a new Synapses class for each
#     defined population-to-population connection, creates the neuron-to-neuron connections
#     as defined by the user (connectivity), the delays and finally initialises any state variables
#     such as the synaptic weights.
#     '''
#     from brian2 import Synapses
#     from brian2 import Equations
#     from scipy.sparse import issparse
#
#     source, target = pops[source], pops[target]
#
#     eqs = synapse['equations']
#     eqs_model = eqs['eqs_model']
#     eqs_pre = eqs['eqs_pre'] if eqs['eqs_pre'] != '' else None
#     eqs_post = eqs['eqs_post'] if eqs['eqs_post'] != '' else None
#     eqs_neuron = eqs['eqs_neuron']
#
#     '''
#     If you define a multiline string for the action to take on a presynaptic spike event, then
#     we split that string and pass it as a list of separate expressions. This way you may define
#     a different delay for each action. See the comment below, near the assignment of delays for
#     more information.
#     '''
#     if eqs_pre and '\n' in eqs_pre:
#         eqs_pre = [eq.strip() for eq in eqs_pre.split('\n')]
#         eqs_pre = [eq for eq in eqs_pre if eq!='']
#
#     model = Equations(eqs_model, **con_info['params'])
#     syn = Synapses(source=source,
#                    target=target,
#                    model=model,
#                    pre=eqs_pre,
#                    post=eqs_post,
#                    clock=clock,
#                    method=solver,
#                    freeze = True, #: Consider using freeze
#                    #compile = True, #Synapses do not support compile yet
#                    code_namespace=con_info['params'])
#     '''
#     Now we have to link any variables that are used in the neuron model, but defined in
#     the synaptic model.
#     This is necessary for the case where we have the dynamics of a variable (say ge)
#     computed in the Synapses, such as: dge/dt = -ge/tau
#     and that variable is used directly in the post-synaptic
#     model, ie: dv/dt = (ge*(v-vsyn) - v) / C
#     Whereby the post-synaptic ge is actually the sum of all the individual ge
#     variables in the Synapses.
#     The linking is done by assigning the Synapse state variable to the NeuronGroup
#     state variable, which would normally look like this:
#     neurons.ge = synapses.ge
#     Unfortunately this has to be done generically in our code, so it looks like the
#     following loop and assumes that the variables to be linked have exactly the same name.
#     '''
#     for lhs in Equations(eqs_neuron)._string:
#         if lhs in model._string:
#             target.__setattr__(lhs, syn.__getattr__(lhs))
#     '''
#     Given synapses we set connectivity and delays between two populations.
#     The simple case is that both are determined by an executable expression or float.
#     Otherwise we get arrays which define connections individually. Due to how
#     the Synapse class works right now, we must pass the indexed array as a string so
#     that the assignment is vectorized.
#     '''
#     connectivity = con_info['connectivity']
#     if issparse(connectivity):
#         connectivity = connectivity.toarray()
#     syn[:,:] = connectivity if isinstance(connectivity,(str,float)) else 'connectivity[i,j]'
#
#     delay = con_info['delays']
#     if issparse(delay):
#         delay = delay.toarray()
#
#     '''
#     If you define more than one action to be taken on a presynaptic spike event, the Synapses
#     class will return a list of SynapticDelayVariable objects. In this case we have
#     to handle each one separately, thus the loop over the list. We assume that if the user
#     has passed an array of predefined delays, that the third axis corresponds to the
#     delay after which each subsequent action is taken.
#     '''
#     syn_delay = syn.delay
#     if isinstance(syn_delay,list):
#         for _k,sd in enumerate(syn_delay):
#             sd[:,:] = delay if isinstance(delay,(str,float)) else 'delay[i,j,_k]'
#     else:
#         syn_delay[:,:] = delay if isinstance(delay,(str,float)) else 'delay[i,j]'
#
#     for varname, svs in con_info['svs']iteritems():
#         if isinstance(svs,(str,float)):
#             set_brian_var(syn, varname, svs)
#         else:
#             var = syn.__getattr__(varname)
#             var[:,:] = 'svs[i,j]'
#
#     return syn

# def run_max(params, neuron_groups, monitor_objs):
#     '''
#     A run implementation that regularly checks population firing rates. If the
#     firing rate of any population with an attached PopulationRateMonitor
#     exceeds 500 Hz, the simulation is stopped early. This function requires at least
#     one PopulationRateMonitor in the monitor_objs['poprate'] dictionary or it will fail.
#     '''
#     from ...utils import filter_network_objs_for_run
#     from brian.network import Network, clear as brian_clear
#     from brian.stdunits import Hz, ms
#     from brian.units import second
#     import numpy as np
#
#     runnable_objs = filter_network_objs_for_run(neuron_groups)
#     network_monitors = filter_network_objs_for_run(monitor_objs)
#     fornetwork = runnable_objs + network_monitors
#     net = Network(*fornetwork)
#
#     maxrate_allowed = 500*Hz
#     min_run_per_iter = 100*ms
#
#     timestep = params['dt']
#     half_timestep = timestep/2.
#
#     popratemons = monitor_objs['poprate'].values()
#     # Find the PopulationRateMonitor with the longest update period, since we can't run any
#     # less than that per iteration of the simulation.
#     pop_rate_step = max([mon._bin for mon in popratemons]) * timestep
#     # If all the monitors are updated more often than every 100 ms, then we will run the
#     # simulation for at least 100 ms per iteration.
#     run_per_iter = max(min_run_per_iter, pop_rate_step)
#     # Figure out how many of the last PopulationRateMonitor.rate values we should average
#     # over to compute the rate in the last run_per_iter time.
#     rate_steps = max(int(run_per_iter / pop_rate_step),1)
#
#     timeremaining = params['runtime']
#     maxrate = 0*Hz
#     while timeremaining > half_timestep and maxrate < maxrate_allowed:
#         run_this_iter = min(timeremaining,run_per_iter)
#         net.run(run_this_iter)
#         timeremaining -= run_this_iter
#         rates = [np.mean(mon.rate[-rate_steps:]) for mon in popratemons]
#         maxrate = max(rates) * Hz
# #        logger.info('Running- {0:.2f}s remain. {1:.2f} Hz poprate'.format(self.timeremaining,
# #                                                                         maxrate))
#     if timeremaining > half_timestep:
#         warn = 'Simulation stopped early: {0:.2f}s remained, {1:.2f} Hz'.format(timeremaining/second,
#                                                                                 maxrate/Hz)
#         print(warn)
#
#     rawdata = brian_monitors_to_rawdata(monitor_objs)
#     brian_clear(True,all=True)
#     return rawdata
# def sort_subpopulations(subpopulations, neuron_groups):
#     sorted_subpops = []
#     for subpopname, subpop_info in iteritems(subpopulations):
#         sps = subpop_info['super']
#         if sps in neuron_groups:
#             sorted_subpops.insert(0,(subpopname,subpop_info))
#         else:
#             found = False
#             for i, subpop in enumerate(sorted_subpops):
#                 if sps == subpop[0]:
#                     found = True
#                     sorted_subpops.insert(i+1,(subpopname,subpop_info))
#             if not found:
#                 sorted_subpops.append((subpopname,subpop_info))
#     return sorted_subpops
# reserved_params = ['reset','threshold','refractory']
#
# def make_population(pt_name, popname, pop_info, models, connections, synapses, clock, solver, subpops):
#     '''
#     Constructs a single population by making a Brian NeuronGroup. No support for adaptive resets
#     or other fanciness. Just makes the Equations, adds synaptic dynamics if appropriate
#     sets the initial conditions and returns the NeuronGroup.
#     '''
#     from brian2 import Equations
#     from brian2 import NeuronGroup
#     from brian2 import ms
#     compileable = ['Euler', 'exponential_Euler', None]
#
#     model = models[pop_info['model']]
#     popparams = pop_info['params']
#
#     refractory  = popparams['refractory'] if 'refractory' in popparams else 0*ms
#     reset = make_reset(pop_info, model, refractory)
#     threshold   = popparams['threshold'] if 'threshold' in popparams else None
#     synaptic_curr_name = model['synaptic_curr_name']
#
#     eqs_params = {k:v for k,v in iteritems(popparams) if k not in reserved_params}
#     eqs = model['equations']
#     eqs = Equations(eqs, **eqs_params)
#
#     eqs += synapses_on_model(connections, synapses, popname, synaptic_curr_name, subpops)
#     ng = NeuronGroup(pop_info['size'], eqs,
#                      threshold = threshold,
#                      refractory = refractory,
#                      reset = reset,
#                      method = solver,
#                      clock = clock,
#                      freeze = True, #: Consider using freeze
#                      compile = solver in compileable, #: Consider using compile
#                      )
#     for var, svs in pop_info['svs'].iteritems():
#         svs = load_state_vars_or_weights(pt_name, svs, (popname, var))
#         set_brian_var(ng, var, svs, pop_info['params'])
#
#     return ng
# def make_subpopulation(pt_name, superpop, subpopname, subpop_info):
#     size = subpop_info['size']
#     params = subpop_info['params']
#     subpop = superpop.subgroup(size)
#
#     for pname, pvalue in [(pname,pvalue) for pname,pvalue in iteritems(params)
#                                             if pname not in reserved_params]:
#         try:
#             subpop.__getattr__(pname)
#             subpop.__setattr__(pname, pvalue)
#         except AttributeError:
#             pass
#
#     for var, svs in subpop_info['svs'].iteritems():
#         svs = load_state_vars_or_weights(pt_name, svs, (subpopname, var))
#         set_brian_var(subpop, var, svs, subpop_info['params'])
#
#     return subpop
#
# def make_reset(pop_info, model, refractory):
#     from brian2 import NoReset, StringReset, Refractoriness, SimpleCustomRefractoriness
#     popparams = pop_info['params']
#     if 'reset_str' in model and model['reset_str'] != '':
#         '''
#         Some annoying hackishness here. Because Brian doesn't allow passing of a namespace
#         into the StringReset or SimpleCustomRefractoriness constructors and Python doesn't
#         allow modification of the locals() dictionary in 2.7+ we have to find a way to get
#         the model parameters into the local namespace before the call to the constructor.
#         This is why we construct a resetlocal dictionary and pass it into eval. Yes it's a
#         hack, but unless you can find a better way to do it, that's how it's going to be.
#         '''
#         resetlocals = {'StringReset':StringReset, 'reset_str':model['reset_str']}
#         resetlocals.update(popparams)
#         reset = eval("StringReset(reset_str)",globals(),resetlocals)
#         if 'refractory' in popparams:
#             reset = SimpleCustomRefractoriness(reset, refractory)
#     elif 'reset' in popparams:
#         reset  = popparams['reset']
#         if 'refractory' in popparams:
#             reset = Refractoriness(reset, refractory)
#     else:
#         reset = NoReset()
#     return reset
#     assert(False)
#
# def synapses_on_model(connections, synapses, popname, synaptic_curr_name, subpops):
#     '''
#     Finds any connections for which this population is postsynaptic. If any are found
#     then we check the synaptic definition to see if the dynamics can be evaluated on the
#     neuron model and if so makes new Equations objects for each synapses.
#     '''
#     from brian2 import Equations
#
#     allpops = [subpopname for subpopname,info in iteritems(subpops) if info['super'] == popname]
#     allpops.append(popname)
#
#     eqs = Equations('')
#     synapses = {connection['synapse']: {'eqs':synapses[connection['synapse']]['equations'],
#                                         'output_var':synapses[connection['synapse']]['output_var'],
#                                         'params':connection['params']}
#                         for connection in itervalues(connections)
#                             if connection['popname_post'] in allpops
#                                 and connection['synapse'] != 'STDP'}
#     allparams = {}
#     for syn_info in itervalues(synapses):
#         allparams.update(syn_info['params'])
#         if syn_info['eqs']['eqs_neuron'] != '':
#             eqs += Equations(syn_info['eqs']['eqs_neuron'], **syn_info['params'])
#
#     if synaptic_curr_name != '':
#         split_syn = synaptic_curr_name.split(':')
#         Isyn_name = split_syn[0]
#         Isyn_units = split_syn[1] if len(split_syn) > 1 else 'amp'
#
#         Isyn = Isyn_name + ' = '
#         if synapses:
#             Isyn += '+'.join(s['output_var'] for s in itervalues(synapses))
#         else:
#             Isyn += '0*'+Isyn_units
#         eqs += Equations(' : '.join((Isyn,Isyn_units)), **allparams)
#
#     return eqs
#
# def make_connection_old(pt_name, pops, source, target, con_info, synapse, clock, solver):
#     #from brian.connections import Connection
#     from scipy.sparse import issparse#, csr_matrix
#     import numpy as np
#
#     eqs = synapse['equations']
#     connectivity = con_info['connectivity']
#     delay = con_info['delays']
#
#     connectivity = load_state_vars_or_weights(pt_name, connectivity, ((source,target),)) #('_'.join((source,target))
#     source, target = pops[source], pops[target]
#
#     delay_issparse = issparse(delay)
#
#     if delay_issparse:
#         max_delay = delay.data.max() if delay.data.size else 0.
#     else:
#         max_delay = np.max(delay)
#
#     scale = np.NaN
#     if 'scale' in con_info['params']:
#         # This is here to that we can vary the connection weights by a constant
#         # factor of 'scale' for each point in paramspace.
#         scale = con_info['params']['scale']
#
# #    TODO: Consider using connect_with_sparse which yields 3x speedup and half the memory usage.
# #     one problem is that the delay matrix must have exactly the same nonzero elements as the
# #     weight matrix, otherwise the network won't run.
#     weight_issparse = issparse(connectivity)
#     if weight_issparse and not (delay_issparse or isinstance(delay,np.ndarray)):
#         # We only handle homogeneous delays right now.
# #        if not delay_issparse:
# #            delay =  csr_matrix(delay)
#         if not np.isnan(scale):
#             connectivity *= scale
#         conn = Connection(source,target,eqs['eqs_model'],delay=delay)
#         conn.connect_from_sparse(connectivity, column_access=True)
#     elif np.isscalar(connectivity):
#         weight = con_info['params']['weight']
#         if not np.isnan(scale):
#             weight *= scale
#         fixed = con_info['params']['fixed'] if 'fixed' in con_info['params'] else False
#         conn = Connection(source,target,eqs['eqs_model'], max_delay=max_delay)
#         if connectivity >= 1.0:
#             conn.connect_full(source, target, weight)
#         else:
#             conn.connect_random(source, target, connectivity, weight, fixed)
#     else:
#         if not np.isnan(scale):
#             connectivity *= scale
#         conn = Connection(source,target,eqs['eqs_model'],
#                           weight=connectivity,
#                           delay=delay,
#                           max_delay=max_delay)
#
#     return conn
#     assert(False)
#
# def make_stdp(stdp_params,conn):
#     from brian.equations import Equations
#     from brian.stdp import STDP
#
#     tau_stdp = stdp_params['tau_stdp']
#     wmax = stdp_params['wmax']
#     wmin = stdp_params['wmin']
#     eta = stdp_params['eta']
#     P = stdp_params['P']
#     D = stdp_params['D']
#     pre = stdp_params['pre']
#     post =stdp_params['post']
#     rho0 = stdp_params['rho0']
#     alpha = rho0*tau_stdp*2
#
#     eqs = stdp_params['eqs']
#
#     params = {'tau_stdp':tau_stdp,
#               'P':P,'D':D,'eta':eta,'alpha':alpha # these not really needed here
#               }
#     eqs_stdp = Equations(eqs, **params)
#     stdp = STDP(conn, eqs=eqs_stdp,
#                    pre=pre,post=post,
#                    wmin=wmin,wmax=wmax)
#     return stdp
#     assert(False)
#
