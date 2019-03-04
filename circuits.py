# V3.19 by psr
#   - follows brian1 code in DB model from Wimmer et al. 2015.
#   - brian2 syntax
#   - runs in cpp_standalone mode
#   - compatible with SNEP
#   - restructured for more readability

import numpy as np
import neuron_models as nm
import get_params as params
from brian2 import PoissonGroup, PoissonInput, linked_var, TimedArray
from brian2.groups import NeuronGroup
from brian2.synapses import Synapses
from brian2.units import amp
from helper_funcs import get_OUstim, unitless


def mk_sen_circuit(task_info):
    """
    Creates balance network representing sensory circuit.

    :return: groups, synapses, subgroups
    """
    # -------------------------------------
    # Params
    # -------------------------------------
    # load params from task_info
    N_E = task_info['sen']['N_E']       # number of exc neurons (1600)
    N_I = task_info['sen']['N_I']       # number of inh neurons (400)
    N_X = task_info['sen']['N_X']       # size of external population
    sub = task_info['sen']['sub']       # fraction of stim-selective exc neurons
    N_E1 = int(N_E * sub)               # size of exc sub-pop 1, 2
    num_method = task_info['sim']['num_method']
    twoComp_model = task_info['sim']['2c_model']

    # define namespace
    if not twoComp_model:
        paramsen = params.get_sen_params(task_info)
    else:
        paramsen = params.get_2c_params(task_info)

    # unpack delays
    dE = paramsen['dE']
    dI = paramsen['dI']
    dX = paramsen['dX']

    # -------------------------------------
    # Set up the model
    # -------------------------------------
    # neuron groups
    if not twoComp_model:
        senE = NeuronGroup(N_E, model=nm.eqs_wimmer_exc, method=num_method, threshold='V>=Vt', reset='V=Vr',
                           refractory='tau_refE', namespace=paramsen, name='senE')
        senE1 = senE[:N_E1]
        senE2 = senE[N_E1:]
    else:
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

    senI = NeuronGroup(N_I, model=nm.eqs_wimmer_inh, method=num_method, threshold='V>=Vt', reset='V=Vr',
                       refractory='tau_refI', namespace=paramsen, name='senI')

    # weight according the different subgroups
    condsame = '(i<N_pre*sub and j<N_post*sub) or (i>=N_pre*sub and j>=N_post*sub)'
    conddiff = '(i<N_pre*sub and j>=N_post*sub) or (i>=N_pre*sub and j<N_post*sub)'

    # -------------------------------------
    # Set up the model and connections
    # -------------------------------------
    # AMPA: exc --> exc
    synSESE = Synapses(senE, senE, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSESE')
    synSESE.connect(p='eps')
    synSESE.w[condsame] = 'w_p * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.w[conddiff] = 'w_m * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.delay = dE

    # AMPA: exc --> inh
    synSESI = Synapses(senE, senI, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                             w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSESI')
    synSESI.connect(p='eps')
    synSESI.w = 'gEI/gleakI * (1 + randn()*0.5)'
    synSESI.delay = dE

    # GABA: inh --> exc
    synSISE = Synapses(senI, senE, model='w : 1', method=num_method,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSISE')
    synSISE.connect(p='eps')
    synSISE.w = 'gIE/gleakE * (1 + randn()*0.5)'
    synSISE.delay = dI

    # GABA: inh --> inh
    synSISI = Synapses(senI, senI, model='w : 1', method=num_method,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSISI')
    synSISI.connect(p='eps')
    synSISI.w = 'gII/gleakI * (1 + randn()*0.5)'
    synSISI.delay = dI

    # external inputs and synapses
    extS = PoissonGroup(N_X, rates='nu_ext', namespace=paramsen)

    synSXSE = Synapses(extS, senE, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSXSE')
    synSXSE.connect(condition=condsame, p='epsX * (1 + alphaX)')
    synSXSE.connect(condition=conddiff, p='epsX * (1 - alphaX)')
    synSXSE.w = 'gXE/gleakE * (1 + randn()*0.5)'
    synSXSE.delay = dX

    synSXSI = Synapses(extS, senI, model='w : 1', method=num_method,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSXSI')
    synSXSI.connect(p='epsX')
    synSXSI.w = 'gXI/gleakI * (1 + randn()*0.5)'
    synSXSI.delay = dX

    # variables to return
    synapses = {'synSESE': synSESE, 'synSESI': synSESI,
                'synSISE': synSISE, 'synSISI': synSISI,
                'synSXSI': synSXSI, 'synSXSE': synSXSE}
    if not twoComp_model:
        groups = {'SE': senE, 'SI': senI, 'SX': extS}
        subgroups = {'SE1': senE1, 'SE2': senE2}
    else:
        groups = {'SE': senE, 'dend': dend, 'SI': senI, 'SX': extS}
        subgroups = {'SE1': senE1, 'SE2': senE2, 'dend1': dend1, 'dend2': dend2}

    return groups, synapses, subgroups


def mk_dec_circuit(task_info):
    """
    Creates the 'winner-takes-all' network described in Wang 2002.

    :return: groups, synapses, update_nmda, subgroups
    """
    # -------------------------------------
    # Params
    # -------------------------------------
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

    # -------------------------------------
    # Set up the model
    # -------------------------------------
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

    # weight according the different subgroups
    condsame = '(label_pre == label_post and label_pre != 3)'
    conddiff = '(label_pre != label_post and label_pre != 3) or (label_pre == 3 and label_post != 3)'
    condrest = '(label_post == 3)'

    # -------------------------------------
    # Connections
    # -------------------------------------
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


def mk_sen_stimulus(task_info, arrays=False):
    """
    Generate common and private part of the stimuli for sensory neurons from an OU process.

    :return: TimedArray with the stimulus for sensory excitatory neurons
    """
    # set seed with np; for standalone mode brian's seed() is not sufficient!
    if task_info['sim']['replicate_stim']:
        # replicated stimuli across iters
        np.random.seed(task_info['seed'])
    else:
        # every iter has different stimuli
        np.random.seed()

    # simulation params
    c = task_info['c']  # stim coherence (between 0 and 1)
    nn = int(task_info['sen']['N_E'] * task_info['sen']['sub'])     # no. of neurons in sub-pop1
    stim_dt = task_info['sim']['stim_dt']
    runtime = unitless(task_info['sim']['runtime'], stim_dt)
    stim_on = unitless(task_info['sim']['stim_on'], stim_dt)
    stim_off = unitless(task_info['sim']['stim_off'], stim_dt)
    stim_dur = stim_off - stim_on
    tp = unitless(stim_dur, stim_dt)  # total points

    # stimulus namespace
    paramstim = params.get_stim_params(task_info)
    tau = unitless(paramstim['tau_stim'], stim_dt)      # OU time constant
    I0 = paramstim['I0']
    mu1 = paramstim['mu1']
    mu2 = paramstim['mu2']
    sigma_stim = paramstim['sigma_stim']
    sigma_ind = paramstim['sigma_ind']
    stim_dt = paramstim['stim_dt']

    # common part
    z1 = np.tile(get_OUstim(tp, tau), (nn, tau))
    z2 = np.tile(get_OUstim(tp, tau), (nn, tau))

    # private part
    zk1 = get_OUstim(tp * nn, tau).reshape(nn, tp)
    zk2 = get_OUstim(tp * nn, tau).reshape(nn, tp)

    # stim2input with zero padding if necessary
    i1 = I0 * (1 + c * mu1 + sigma_stim * z1 + sigma_ind * zk1)
    i2 = I0 * (1 + c * mu2 + sigma_stim * z2 + sigma_ind * zk2)
    i1t = np.concatenate((np.zeros((stim_on, nn)), i1.T, np.zeros((runtime - stim_off, nn))), axis=0)
    i2t = np.concatenate((np.zeros((stim_on, nn)), i2.T, np.zeros((runtime - stim_off, nn))), axis=0)
    Irec = TimedArray(np.concatenate((i1t, i2t), axis=1)*amp, dt=stim_dt)

    if arrays:
        stim1 = i1t.T
        stim2 = i2t.T
        stim_time = np.arange(0, runtime, stim_dt)
        return Irec, stim1, stim2, stim_time

    return Irec


def mk_fffb_synapses(task_info, dec_subgroups, sen_subgroups):
    """
    Feedforward and feedback synapses from sensory to integration.

    :return: dictionary with two synapse objects
    """
    # params
    num_method = task_info['sim']['num_method']
    paramfffb = params.get_fffb_params(task_info)
    d = paramfffb['d']

    # unpack subgroups
    decE1 = dec_subgroups['DE1']
    decE2 = dec_subgroups['DE2']
    senE1 = sen_subgroups['SE1']
    senE2 = sen_subgroups['SE2']
    if not task_info['sim']['2c_model']:
        fb_target1 = senE1
        fb_target2 = senE2
    else:
        dend1 = sen_subgroups['dend1']
        dend2 = sen_subgroups['dend2']
        fb_target1 = dend1
        fb_target2 = dend2

    # create feedforward synapses
    synSE1DE1 = Synapses(senE1, decE1, model='w = w_ff : 1', method=num_method,
                         on_pre='g_ea += w', delay=d, name='synSE1DE1', namespace=paramfffb)
    synSE2DE2 = Synapses(senE2, decE2, model='w = w_ff : 1', method=num_method,
                         on_pre='g_ea += w', delay=d, name='synSE2DE2', namespace=paramfffb)

    # create feedback synapses
    synDE1SE1 = Synapses(decE1, fb_target1, model='w = w_fb : 1', method=num_method,
                         on_pre='x_ea += w', delay=d, name='synDE1SE1', namespace=paramfffb)
    synDE2SE2 = Synapses(decE2, fb_target2, model='w = w_fb : 1', method=num_method,
                         on_pre='x_ea += w', delay=d, name='synDE2SE2', namespace=paramfffb)

    # connect synapses
    for syn in [synSE1DE1, synSE2DE2, synDE1SE1, synDE2SE2]:
        syn.connect(p='eps')

    fffb_synapses = {'synSE1DE1': synSE1DE1, 'synSE2DE2': synSE2DE2,
                     'synDE1SE1': synDE1SE1, 'synDE2SE2': synDE2SE2}

    return fffb_synapses


def set_init_conds(task_info, dec_groups, sen_groups):
    """
    Set the adequate initial conditions for the simulation.

    :return: the initialized neuron groups
    """
    # initial conditions
    for dec_group in dec_groups:
        try:
            # init near Vt instead of 0 to avoid initial bump!
            dec_group.V = '-52*mV + 2*mV*rand()'
            dec_group.g_ea = '0.05 * (1 + 0.2*rand())'
        except KeyError:
            # ignore external population
            pass
    for sen_group in sen_groups:
        try:
            sen_group.V = '-52*mV + 2*mV * rand()'
            sen_group.g_ea = '0.2 * rand()'
        except KeyError:
            pass

    if task_info['sim']['2c_model']:
        last_muOUd = np.loadtxt('last_muOUd.csv')
        sen_groups['dend'].g_ea = '0.05 * (1 + 0.2*rand())'
        sen_groups['dend'].V_d = '-72*mV + 2*mV*rand()'
        sen_groups['dend'].muOUd = np.tile(last_muOUd, 2) * amp

    return dec_groups, sen_groups


def mk_monitors(task_info, dec_subgroups, sen_subgroups, dec_groups, sen_groups):
    """
    Define the monitors to track simulation results.

    :return: a list containing the defined monitors
    """
    # import monitors from brian
    from brian2.monitors import SpikeMonitor, PopulationRateMonitor

    # unpack subgroups
    decE1 = dec_subgroups['DE1']
    decE2 = dec_subgroups['DE2']
    senE1 = sen_subgroups['SE1']
    senE2 = sen_subgroups['SE2']
    senE = sen_groups['SE']

    # create monitors
    nnSE = int(task_info['sen']['N_E'] * task_info['sen']['sub'])     # no. of neurons in sub-pop1
    nn2rec = int(100)                                                 # no. of neurons to record
    spksSE = SpikeMonitor(senE[nnSE-nn2rec:nnSE+nn2rec])              # lasts of SE1 and firsts SE2
    rateDE1 = PopulationRateMonitor(decE1)
    rateDE2 = PopulationRateMonitor(decE2)
    rateSE1 = PopulationRateMonitor(senE1)
    rateSE2 = PopulationRateMonitor(senE2)

    if task_info['sim']['plt_fig1']:
        # spk monitors
        spksSE = SpikeMonitor(senE)
        decE = dec_groups['DE']
        nnDE = int(task_info['dec']['N_E'] * 2 * task_info['dec']['sub'])
        spksDE = SpikeMonitor(decE[:nnDE])
        rateDI = PopulationRateMonitor(dec_groups['DI'])
        rateSI = PopulationRateMonitor(sen_groups['SI'])

        return [spksSE, spksDE, rateDE1, rateDE2, rateDI, rateSE1, rateSE2, rateSI]

    return [spksSE, rateDE1, rateDE2, rateSE1, rateSE2]

