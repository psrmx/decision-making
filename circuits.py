# V3.19 by psr
#   - follows code in DB model from Wimmer et al. 2015.
#   - brian2 syntax
#   - uses in cpp_standalone mode
#   - compatible with SNEP
#   - restructured for more readability

import numpy as np
import neuron_models as nm
import get_params as params
from brian2 import PoissonGroup, PoissonInput, linked_var, TimedArray, seed, Network
from brian2.groups import NeuronGroup
from brian2.synapses import Synapses
from brian2.units import amp, ms
from helper_funcs import get_OUstim, unitless


def mk_dec_circuit(task_info):
    """
    Creates the 'winner-takes-all' network described in Wang 2002.

    :return: groups, synapses, update_nmda, subgroups
    """
    # load params from task_info
    N_E = task_info['dec']['N_E']       # number of exc neurons (1600)
    N_I = task_info['dec']['N_I']       # number of inh neurons (400)
    sub = task_info['dec']['sub']       # fraction of stim-selective exc neurons
    N_D1 = int(N_E * sub)               # size of exc sub-pop D1
    N_D2 = N_D1                         # size of exc sub-pop D2
    N_D3 = int(N_E * (1 - 2 * sub))     # size of exc sub-pop D3, the rest
    num_method = task_info['sim']['num_method']

    # define namespace
    paramdec = params.get_dec_params(task_info)

    # unpack variables
    d = paramdec['d']
    nu_ext = paramdec['nu_ext']
    nu_ext1 = paramdec['nu_ext1']

    # neuron groups
    decE = NeuronGroup(N_E, model=nm.eqs_wang_exc, method=num_method, threshold='V>=Vt', reset='V=Vr',
                       refractory='tau_refE', namespace=paramdec, name='decE')
    decE1 = decE[:N_D1]
    decE2 = decE[N_D1:N_D1 + N_D2]
    decE3 = decE[-N_D3:]
    decE1.label = 1
    decE2.label = 2
    decE3.label = 3

    decI = NeuronGroup(N_I, model=nm.eqs_wang_inh, method=num_method, threshold='V>=Vt', reset='V=Vr',
                       refractory='tau_refI', namespace=paramdec, name='decI')

    # weight connections according to different subgroups
    condsame = '(label_pre == label_post and label_pre != 3)'
    conddiff = '(label_pre != label_post and label_pre != 3) or (label_pre == 3 and label_post != 3)'
    condrest = '(label_post == 3)'

    # NMDA: exc --> exc
    synDEDEn = Synapses(decE, decE, model=nm.eqs_NMDA, method=num_method,
                        on_pre='x_en += w', delay=d,
                        namespace=paramdec, name='synDEDEn')
    synDEDEn.connect()
    synDEDEn.w['i == j'] = 1
    synDEDEn.w['i != j'] = 0
    synDEDEn.w_nmda[condsame] = 'w_p * gEEn/gleakE'
    synDEDEn.w_nmda[conddiff] = 'w_m * gEEn/gleakE'
    synDEDEn.w_nmda[condrest] = 'gEEn/gleakE'

    # NMDA: exc --> inh
    decI.w_nmda = '(gEIn/gleakI) / (gEEn/gleakE)'
    decI.g_ent = linked_var(decE3, 'g_ent', index=range(N_I))

    # AMPA: exc --> exc
    synDEDEa = Synapses(decE, decE, model='w : 1', method=num_method,
                        on_pre='g_ea += w', delay=d,
                        namespace=paramdec, name='synDEDEa')
    synDEDEa.connect()
    synDEDEa.w[condsame] = 'w_p * gEEa/gleakE'
    synDEDEa.w[conddiff] = 'w_m * gEEa/gleakE'
    synDEDEa.w[condrest] = 'gEEa/gleakE'

    # AMPA: exc --> inh
    synDEDIa = Synapses(decE, decI, model='w : 1', method=num_method,
                        on_pre='g_ea += w', delay=d,
                        namespace=paramdec, name='synDEDIa')
    synDEDIa.connect()
    synDEDIa.w = 'gEIa/gleakI'

    # GABA: inh --> exc
    synDIDE = Synapses(decI, decE, model='w : 1', method=num_method,
                       on_pre='g_i += w', delay=d,
                       namespace=paramdec, name='synDIDE')
    synDIDE.connect()
    synDIDE.w = 'gIE/gleakE'

    # GABA: inh --> inh
    synDIDI = Synapses(decI, decI, model='w : 1', method=num_method,
                       on_pre='g_i += w', delay=d,
                       namespace=paramdec, name='synDIDI')
    synDIDI.connect()
    synDIDI.w = 'gII/gleakI'

    # external inputs
    extE = PoissonInput(decE[:N_D1 + N_D2], 'g_ea', N=1, rate=nu_ext1, weight='gXE/gleakE')
    extE3 = PoissonInput(decE3, 'g_ea', N=1, rate=nu_ext, weight='gXE/gleakE')
    extI = PoissonInput(decI, 'g_ea', N=1, rate=nu_ext, weight='gXI/gleakI')

    # variables to return
    groups = {'DE': decE, 'DI': decI, 'DX': extE, 'DX3': extE3, 'DXI': extI}
    subgroups = {'DE1': decE1, 'DE2': decE2, 'DE3': decE3}
    synapses = {'synDEDEn': synDEDEn, 'synDEDEa': synDEDEa, 'synDEDIa': synDEDIa,
                'synDIDE': synDIDE, 'synDIDI': synDIDI}

    return groups, synapses, subgroups


def mk_sen_circuit(task_info):
    """
    Creates balance network representing sensory circuit.

    :return: groups, synapses, subgroups
    """
    # load params from task_info
    N_E = task_info['sen']['N_E']       # number of exc neurons (1600)
    N_I = task_info['sen']['N_I']       # number of inh neurons (400)
    N_X = task_info['sen']['N_X']       # size of external population
    sub = task_info['sen']['sub']       # fraction of stim-selective exc neurons
    N_E1 = int(N_E * sub)               # size of exc sub-pop 1, 2
    num_method = task_info['sim']['num_method']
    two_comp = task_info['sim']['2c_model']

    # define namespace
    if two_comp:
        paramsen = params.get_2c_params(task_info)
    else:
        paramsen = params.get_sen_params(task_info)

    # neuron groups
    if two_comp:
        senE = NeuronGroup(N_E, model=nm.eqs_naud_soma, method=num_method, threshold='V>=Vt',
                           reset='''V = Vl
                                    w_s += bws''',
                           refractory='tau_refE', namespace=paramsen, name='senE')
        dend = NeuronGroup(N_E, model=nm.eqs_naud_dend, method=num_method, namespace=paramsen, name='dend')
        senE.V_d = linked_var(dend, 'V_d')
        dend.lastspike_soma = linked_var(senE, 'lastspike')
        senE1 = senE[:N_E1]
        senE2 = senE[N_E1:]
        dend1 = dend[:N_E1]
        dend2 = dend[N_E1:]
    else:
        senE = NeuronGroup(N_E, model=nm.eqs_wimmer_exc, method=num_method, threshold='V>=Vt', reset='V=Vr',
                           refractory='tau_refE', namespace=paramsen, name='senE')
        senE1 = senE[:N_E1]
        senE2 = senE[N_E1:]

    senI = NeuronGroup(N_I, model=nm.eqs_wimmer_inh, method=num_method, threshold='V>=Vt', reset='V=Vr',
                       refractory='tau_refI', namespace=paramsen, name='senI')
    extS = PoissonGroup(N_X, rates='nu_ext', namespace=paramsen)

    # variables to return
    synapses = mk_sen_synapses(task_info, senE, senI, extS, paramsen)

    if two_comp:
        groups = {'SE': senE, 'dend': dend, 'SI': senI, 'SX': extS}
        subgroups = {'SE1': senE1, 'SE2': senE2, 'dend1': dend1, 'dend2': dend2}
    else:
        groups = {'SE': senE, 'SI': senI, 'SX': extS}
        subgroups = {'SE1': senE1, 'SE2': senE2}

    return groups, synapses, subgroups


def mk_sen_circuit_plastic(task_info):
    """
    Creates sensory circuit with inhibitory plasticity acting on dendrites.

    :return: groups, synapses, subgroups
    """
    # load params from task_info
    N_E = task_info['sen']['N_E']       # number of exc neurons (1600)
    N_I = task_info['sen']['N_I']       # number of inh neurons (400)
    N_X = task_info['sen']['N_X']       # size of external population
    sub = task_info['sen']['sub']       # fraction of stim-selective exc neurons
    N_E1 = int(N_E * sub)               # size of exc sub-pop 1, 2
    num_method = task_info['sim']['num_method']

    # define namespace
    paramplastic = params.get_plasticity_params(task_info)
    tau_update = paramplastic['tau_update']

    # neuron groups
    eqs_soma_plastic = nm.eqs_naud_soma + nm.eqs_plasticity_linked
    eqs_dend_plastic = nm.eqs_naud_dend + nm.eqs_plasticity
    senE = NeuronGroup(N_E, model=eqs_soma_plastic, method=num_method, threshold='V>=Vt',
                       reset='''V = Vl
                                w_s += bws
                                burst_start += 1
                                burst_stop = 1''',
                       refractory='tau_refE', namespace=paramplastic, name='senE')
    dend = NeuronGroup(N_E, model=eqs_dend_plastic, method=num_method, threshold='burst_start > 1 + min_burst_stop',
                       reset='''B += 1
                                burst_start = 0''',
                       refractory='burst_stop >= min_burst_stop', namespace=paramplastic, name='dend')
    senI = NeuronGroup(N_I, model=nm.eqs_wimmer_inh, method=num_method, threshold='V>=Vt', reset='V=Vr',
                       refractory='tau_refI', namespace=paramplastic, name='senI')
    extS = PoissonGroup(N_X, rates='nu_ext', namespace=paramplastic)

    # linked variables
    senE.V_d = linked_var(dend, 'V_d')
    senE.B = linked_var(dend, 'B')
    senE.burst_start = linked_var(dend, 'burst_start')
    senE.burst_stop = linked_var(dend, 'burst_stop')
    senE.muOUd = linked_var(dend, 'muOUd')
    dend.lastspike_soma = linked_var(senE, 'lastspike')

    # subgroups
    senE1 = senE[:N_E1]
    senE2 = senE[N_E1:]
    dend1 = dend[:N_E1]
    dend2 = dend[N_E1:]

    # update rule
    dend1.muOUd = '-50*pA - rand()*100*pA'  # random initialisation in [-150:-50 pA]
    dend1.run_regularly('muOUd = clip(muOUd - eta * (B - B0), -100*amp, 0)', dt=tau_update)

    # connections
    sen_synapses = mk_sen_synapses(task_info, senE, senI, extS, paramplastic)
    extD, synDXdend = mk_poisson_fb(task_info, dend)

    # variables to return
    groups = {'SE': senE, 'dend': dend, 'SI': senI, 'SX': extS, 'DX': extD}
    subgroups = {'SE1': senE1, 'SE2': senE2,
                 'dend1': dend1, 'dend2': dend2}
    synapses = {**sen_synapses, **{'synDXdend': synDXdend}}

    return groups, synapses, subgroups


def mk_sen_synapses(task_info, exc, inh, ext, param_space):
    """creates synapses for the different types of sensory circuits"""
    # unpack variables
    num_method = task_info['sim']['num_method']
    sub = task_info['sen']['sub']
    dE = param_space['dE']
    dI = param_space['dI']
    dX = param_space['dX']

    # weight according to different subgroups
    condsame = '(i<N_pre*sub and j<N_post*sub) or (i>=N_pre*sub and j>=N_post*sub)'
    conddiff = '(i<N_pre*sub and j>=N_post*sub) or (i>=N_pre*sub and j<N_post*sub)'

    # AMPA: exc --> exc
    synSESE = Synapses(exc, exc, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSESE')
    synSESE.connect(p='eps')
    synSESE.w[condsame] = 'w_p * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.w[conddiff] = 'w_m * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.delay = dE

    # AMPA: exc --> inh
    synSESI = Synapses(exc, inh, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSESI')
    synSESI.connect(p='eps')
    synSESI.w = 'gEI/gleakI * (1 + randn()*0.5)'
    synSESI.delay = dE

    # GABA: inh --> exc
    synSISE = Synapses(inh, exc, model='w : 1', method=num_method,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSISE')
    synSISE.connect(p='eps')
    synSISE.w = 'gIE/gleakE * (1 + randn()*0.5)'
    synSISE.delay = dI

    # GABA: inh --> inh
    synSISI = Synapses(inh, inh, model='w : 1', method=num_method,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSISI')
    synSISI.connect(p='eps')
    synSISI.w = 'gII/gleakI * (1 + randn()*0.5)'
    synSISI.delay = dI

    # external inputs and synapses
    synSXSE = Synapses(ext, exc, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSXSE')
    synSXSE.connect(condition=condsame, p='epsX * (1 + alphaX)')
    synSXSE.connect(condition=conddiff, p='epsX * (1 - alphaX)')
    synSXSE.w = 'gXE/gleakE * (1 + randn()*0.5)'
    synSXSE.delay = dX

    synSXSI = Synapses(ext, inh, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param_space, name='synSXSI')
    synSXSI.connect(p='epsX')
    synSXSI.w = 'gXI/gleakI * (1 + randn()*0.5)'
    synSXSI.delay = dX

    # variables to return
    synapses = {'synSESE': synSESE, 'synSESI': synSESI,
                'synSISE': synSISE, 'synSISI': synSISI,
                'synSXSE': synSXSE, 'synSXSI': synSXSI}

    return synapses


def mk_sen_stimulus(task_info, arrays=False):
    """
    Generate common and private part of the stimuli for sensory neurons from an OU process.

    :return: TimedArray with the stimulus for sensory excitatory neurons
    """
    # set seed with np - for standalone mode brian's seed() is not sufficient!
    if task_info['sim']['replicate_stim']:
        # replicated stimuli across iters
        np.random.seed(task_info['seed'])
    else:
        # every iter has different stimuli
        np.random.seed()

    # simulation params
    nn = int(task_info['sen']['N_E'] * task_info['sen']['sub'])     # no. of neurons in sub-pop1
    stim_dt = task_info['sim']['stim_dt']
    runtime = unitless(task_info['sim']['runtime'], stim_dt)
    stim_on = unitless(task_info['sim']['stim_on'], stim_dt)
    stim_off = unitless(task_info['sim']['stim_off'], stim_dt)
    tp = stim_off - stim_on                             # total stim points

    # stimulus namespace
    paramstim = params.get_stim_params(task_info)
    tau = unitless(paramstim['tau_stim'], stim_dt)      # OU time constant
    c = paramstim['c']
    I0 = paramstim['I0']
    mu1 = paramstim['mu1']
    mu2 = paramstim['mu2']
    sigma_stim = paramstim['sigma_stim']
    sigma_ind = paramstim['sigma_ind']

    # common and private part
    z1 = np.tile(get_OUstim(tp, tau), (nn, 1))
    z2 = np.tile(get_OUstim(tp, tau), (nn, 1))
    zk1 = get_OUstim(tp * nn, tau).reshape(nn, tp)
    zk2 = get_OUstim(tp * nn, tau).reshape(nn, tp)

    # stim2TimedArray with zero padding if necessary
    i1 = I0 * (1 + c * mu1 + sigma_stim * z1 + sigma_ind * zk1)
    i2 = I0 * (1 + c * mu2 + sigma_stim * z2 + sigma_ind * zk2)
    i1t = np.concatenate((np.zeros((stim_on, nn)), i1.T, np.zeros((runtime - stim_off, nn))), axis=0)
    i2t = np.concatenate((np.zeros((stim_on, nn)), i2.T, np.zeros((runtime - stim_off, nn))), axis=0)
    Irec = TimedArray(np.concatenate((i1t, i2t), axis=1)*amp, dt=stim_dt)

    if arrays:
        stim1 = i1t.T
        stim2 = i2t.T
        stim_time = np.linspace(0, task_info['sim']['runtime'], runtime)
        return Irec, stim1, stim2, stim_time

    return Irec


def mk_fffb_synapses(task_info, dec_subgroups, sen_subgroups):
    """
    Feedforward and feedback synapses of hierarchical network.

    :return: dictionary with the synapses objects
    """
    # params
    paramfffb = params.get_fffb_params(task_info)
    d = paramfffb['d']
    num_method = task_info['sim']['num_method']
    two_comp = task_info['sim']['2c_model']

    # unpack subgroups
    decE1 = dec_subgroups['DE1']
    decE2 = dec_subgroups['DE2']
    senE1 = sen_subgroups['SE1']
    senE2 = sen_subgroups['SE2']
    if not two_comp:
        fb_target1 = senE1
        fb_target2 = senE2
    else:
        fb_target1 = sen_subgroups['dend1']
        fb_target2 = sen_subgroups['dend2']

    # create FF and FB synapses
    synSE1DE1 = Synapses(senE1, decE1, model='w = w_ff : 1', method=num_method,
                         on_pre='g_ea += w', delay=d, name='synSE1DE1', namespace=paramfffb)
    synSE2DE2 = Synapses(senE2, decE2, model='w = w_ff : 1', method=num_method,
                         on_pre='g_ea += w', delay=d, name='synSE2DE2', namespace=paramfffb)
    synDE1SE1 = Synapses(decE1, fb_target1, model='w = w_fb : 1', method=num_method,
                         on_pre='x_ea += w', delay=d, name='synDE1SE1', namespace=paramfffb)
    synDE2SE2 = Synapses(decE2, fb_target2, model='w = w_fb : 1', method=num_method,
                         on_pre='x_ea += w', delay=d, name='synDE2SE2', namespace=paramfffb)
    for syn in [synSE1DE1, synSE2DE2, synDE1SE1, synDE2SE2]:
        syn.connect(p='eps')

    fffb_synapses = {'synSE1DE1': synSE1DE1, 'synSE2DE2': synSE2DE2,
                     'synDE1SE1': synDE1SE1, 'synDE2SE2': synDE2SE2}

    return fffb_synapses


def mk_poisson_fb(task_info, dend):
    """
    Feedback synapses from poisson mimicking decision circuit, to sensory.

    :return: a poisson group and a synapse object
    """
    # params
    paramfffb = params.get_fffb_params(task_info)
    d = paramfffb['d']
    num_method = task_info['sim']['num_method']

    # Poisson group
    N_E = task_info['sen']['N_E']           # number of exc neurons (1600)
    subDE = task_info['dec']['sub']         # stim-selective fraction in decision exc neurons
    N_DX = int(subDE * N_E)                 # number decision mock neurons
    extD = PoissonGroup(N_DX, rates=task_info['plastic']['dec_winner_rate'])

    # FB synapse
    synDXdend = Synapses(extD, dend, model='w = w_fb : 1', method=num_method, delay=d,
                         on_pre='x_ea += w',
                         namespace=paramfffb, name='synXDEdend')
    synDXdend.connect(p='eps')

    return extD, synDXdend


def set_init_conds(neuron_groups, two_comp=False, plastic=False):
    """Returns the initialized neuron_groups according to adequate values."""
    for neuron_group in neuron_groups.values():
        if not neuron_group.name.startswith(('poisson', 'dend')):
            try:
                # init near Vt instead of 0 to avoid initial bump!
                neuron_group.V = '-52*mV + 2*mV * rand()'
                neuron_group.g_ea = '0.2 * rand()'
            except AttributeError:
                pass

    if two_comp:
        neuron_groups['dend'].V_d = '-72*mV + 2*mV*rand()'
        neuron_groups['dend'].g_ea = '0.2*rand()'
        if not plastic:
            last_muOUd = np.loadtxt('last_muOUd.csv')
            neuron_groups['dend'].muOUd = np.tile(last_muOUd, 2) * amp

    return neuron_groups


def mk_monitors(task_info, dec_groups, sen_groups, dec_subgroups, sen_subgroups):
    """Define monitors to track results from hierarchical experiment."""
    from brian2.monitors import SpikeMonitor, PopulationRateMonitor

    # unpack neuron groups
    senE = sen_groups['SE']
    decE1 = dec_subgroups['DE1']
    decE2 = dec_subgroups['DE2']
    senE1 = sen_subgroups['SE1']
    senE2 = sen_subgroups['SE2']

    # create monitors
    nnSE = int(task_info['sen']['N_E'] * task_info['sen']['sub'])     # no. of neurons in sub-pop1
    nn2rec = int(100)                                                 # no. of neurons to record
    spksSE = SpikeMonitor(senE[nnSE-nn2rec:nnSE+nn2rec])              # lasts of SE1 and firsts SE2
    rateDE1 = PopulationRateMonitor(decE1)
    rateDE2 = PopulationRateMonitor(decE2)
    rateSE1 = PopulationRateMonitor(senE1)
    rateSE2 = PopulationRateMonitor(senE2)

    monitors = [spksSE, rateDE1, rateDE2, rateSE1, rateSE2]

    if task_info['sim']['plt_fig1']:
        spksSE = SpikeMonitor(senE)
        decE = dec_groups['DE']
        nnDE = int(task_info['dec']['N_E'] * 2 * task_info['dec']['sub'])
        spksDE = SpikeMonitor(decE[:nnDE])
        rateDI = PopulationRateMonitor(dec_groups['DI'])
        rateSI = PopulationRateMonitor(sen_groups['SI'])
        monitors = [spksSE, spksDE, rateDE1, rateDE2, rateDI, rateSE1, rateSE2, rateSI]

    return monitors


def mk_monitors_plastic(sen_groups, sen_subgroups):
    """Define monitors to track results from plasticity experiment."""
    from brian2.monitors import SpikeMonitor, StateMonitor

    # unpack neuron groups
    senE = sen_groups['SE']
    dend1 = sen_subgroups['dend1']

    # create monitors
    spksSE = SpikeMonitor(senE)
    dend_mon = StateMonitor(dend1, variables=['muOUd', 'Ibg', 'g_ea', 'B'], record=True, dt=1*ms)

    return [spksSE, dend_mon]


def get_hierarchical_net(task_info):
    """
    Construct hierarchical net for decision making experiment.

    :return: Brian Network object to run and monitors for plotting
    """
    dec_groups, dec_synapses, dec_subgroups = mk_dec_circuit(task_info)
    sen_groups, sen_synapses, sen_subgroups = mk_sen_circuit(task_info)
    fffb_synapses = mk_fffb_synapses(task_info, dec_subgroups, sen_subgroups)

    seed()
    dec_groups = set_init_conds(dec_groups)
    sen_groups = set_init_conds(sen_groups, two_comp=task_info['sim']['2c_model'])
    monitors = mk_monitors(task_info, dec_groups, sen_groups, dec_subgroups, sen_subgroups)
    net = Network(dec_groups.values(), dec_synapses.values(),
                  sen_groups.values(), sen_synapses.values(),
                  fffb_synapses.values(), *monitors, name='hierarchicalnet')

    return net, monitors


def get_plasticity_net(task_info):
    """Construct sensory circuit for inhibitory plasticity experiment."""
    sen_groups, sen_synapses, sen_subgroups = mk_sen_circuit_plastic(task_info)

    seed()
    sen_groups = set_init_conds(sen_groups, two_comp=True, plastic=True)
    monitors = mk_monitors_plastic(sen_groups, sen_subgroups)
    net = Network(sen_groups.values(), sen_synapses.values(), *monitors, name='plasticitynet')

    return net, monitors
