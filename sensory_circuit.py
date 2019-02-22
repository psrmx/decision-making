from brian2 import PoissonGroup, linked_var
from brian2.groups import NeuronGroup
from brian2.synapses import Synapses
from brian2.units import Hz, ms, second

def mk_sencircuit(task_info):
    """
    Creates balanced network representing sensory circuit.

    returns:
        groups, synapses, subgroups

    groups and synapses have to be added to the "Network" in order to run the simulation
    subgroups is used for establishing connections between the sensory and integration circuit;
    do not add subgroups to the "Network"


    psr following the published Brian1 code in DB model from Wimmer et al. 2015
    - brain2 code
    - implementation runs in cpp_standalone mode, compatible with SNEP
    """
    # -------------------------------------
    # Sensory circuit parameters
    # -------------------------------------
    # populations
    N_E = task_info['sen']['populations']['N_E']  # total number of E neurons
    sub = task_info['sen']['populations']['sub']  # fraction corresponding to subpops 1,2
    N_E1 = int(N_E * sub)  # size of exc subpops that responds to the stimuli
    N_I = task_info['sen']['populations']['N_I']  # total number of I neurons
    N_X = task_info['sen']['populations']['N_X']  # size of external pop

    # local recurrent connections
    eps = task_info['sen']['connectivity']['eps']  # connection probability for EE, EI, IE and II
    w_p = task_info['sen']['connectivity']['w_p']  # relative synaptic strength within pop E1 and E2
    w_m = 2 - w_p  # relative synaptic strength across pop E1 and E2
    gEE = task_info['sen']['connectivity']['gEE']  # weight of EE synapses, prev: 0.7589*nS
    gEI = task_info['sen']['connectivity']['gEI']  # weight of EI synapses, prev: 1.5179*nS
    gIE = task_info['sen']['connectivity']['gIE']  # weight of IE synapses, prev: 12.6491*nS
    gII = task_info['sen']['connectivity']['gII']  # weight of II synapses
    gmax = task_info['sen']['connectivity']['gmax']  # maximum synaptic weight
    dE = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of E synapses, (0.5:1.5)
    dI = '0.1*ms + rand()*0.8*ms'  # range of transmission delays of I synapses, (0.1:0.9)

    # external connections
    epsX = task_info['sen']['connectivity']['epsX']  # connection probability for ext synapses
    alphaX = task_info['sen']['connectivity']['alphaX']  # 0: global input, 1: local input
    gXE = task_info['sen']['connectivity']['gXE']  # weight of ext to E synapses
    gXI = task_info['sen']['connectivity']['gXI']  # weight of ext to I synapses
    dX = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of X synapses, (0.5:1.5)

    # neuron models
    CmE = task_info['sen']['neuron']['CmE']  # membrane capacitance of E neurons
    CmI = task_info['sen']['neuron']['CmI']  # membrane capacitance of I neurons
    gleakE = task_info['sen']['neuron']['gleakE']  # leak conductance of E neurons
    gleakI = task_info['sen']['neuron']['gleakI']  # leak conductance of I neurons
    Vl = task_info['sen']['neuron']['Vl']  # resting potential
    Vt = task_info['sen']['neuron']['Vt']  # spiking threshold
    Vr = task_info['sen']['neuron']['Vr']  # reset potential
    tau_refE = task_info['sen']['neuron']['tau_refE']  # absolute refractory period of E neurons
    tau_refI = task_info['sen']['neuron']['tau_refI']  # absolute refractory period of I neurons
    nu_ext = task_info['sen']['neuron']['nu_ext']  # rate of external Poisson neurons, prev: 12.5*Hz

    # synapse models
    VrevE = task_info['sen']['synapse']['VrevE']  # reversal potential for E synapses
    VrevI = task_info['sen']['synapse']['VrevI']  # reversal potential for I synapses
    tau_decay = task_info['sen']['synapse']['tau_decay']  # decay constants of AMPA and GABA conductances
    tau_rise = task_info['sen']['synapse']['tau_rise']  # rise constants of AMPA and GABA conductances

    # namespace with params
    paramsen = {'gEE': gEE, 'gEI': gEI, 'gIE': gIE, 'gII': gII, 'gmax': gmax, 'gXE': gXE, 'gXI': gXI,
                'gleakE': gleakE, 'gleakI': gleakI, 'CmE': CmE, 'CmI': CmI, 'Vl': Vl, 'Vt': Vt, 'Vr': Vr,
                'VrevE': VrevE, 'VrevI': VrevI, 'tau_decay': tau_decay, 'tau_rise': tau_rise,
                'w_p': w_p, 'w_m': w_m, 'sub': sub, 'eps': eps, 'epsX': epsX, 'alphaX': alphaX}

    # numerical integration method
    nummethod = task_info['simulation']['nummethod']

    # -------------------------------------
    # Set up the model and connections
    # -------------------------------------
    # neuron equations
    eqsE = '''
        dV/dt = (-g_ea*(V-VrevE) - g_i*(V-VrevI) - (V-Vl)) / tau  + I/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt  = -x_ea / tau_rise            : 1
        dg_i/dt  = (-g_i + x_i) / tau_decay     : 1
        dx_i/dt  = -x_i / tau_rise              : 1
        tau = CmE/gleakE    : second
        Cm = CmE            : farad
        I = Irec(t, i)      : amp
    '''

    eqsI = '''
        dV/dt = (-g_ea*(V-VrevE) - g_i*(V-VrevI) - (V-Vl)) / tau  + I/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt  = -x_ea / tau_rise            : 1
        dg_i/dt  = (-g_i + x_i) / tau_decay     : 1
        dx_i/dt  = -x_i / tau_rise              : 1
        tau = CmI/gleakI    : second
        Cm = CmI            : farad
        I : amp
    '''

    # neuron groups
    senE = NeuronGroup(N_E, model=eqsE, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refE, namespace=paramsen, name='senE')
    senE1 = senE[:N_E1]
    senE2 = senE[N_E1:]

    senI = NeuronGroup(N_I, model=eqsI, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refI, namespace=paramsen, name='senI')

    # synapses
    # weight according the different subgroups
    condsame = '(i<N_pre*sub and j<N_post*sub) or (i>=N_pre*sub and j>=N_post*sub)'
    conddiff = '(i<N_pre*sub and j>=N_post*sub) or (i>=N_pre*sub and j<N_post*sub)'

    # AMPA: exc --> exc
    synSESE = Synapses(senE, senE, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSESE')
    synSESE.connect(p='eps')
    synSESE.w[condsame] = 'w_p * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.w[conddiff] = 'w_m * gEE/gleakE * (1 + randn()*0.5)'
    synSESE.delay = dE

    # AMPA: exc --> inh
    synSESI = Synapses(senE, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSESI')
    synSESI.connect(p='eps')
    synSESI.w = 'gEI/gleakI * (1 + randn()*0.5)'
    synSESI.delay = dE

    # GABA: inh --> exc
    synSISE = Synapses(senI, senE, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSISE')
    synSISE.connect(p='eps')
    synSISE.w = 'gIE/gleakE * (1 + randn()*0.5)'
    synSISE.delay = dI

    # GABA: inh --> inh
    synSISI = Synapses(senI, senI, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSISI')
    synSISI.connect(p='eps')
    synSISI.w = 'gII/gleakI * (1 + randn()*0.5)'
    synSISI.delay = dI

    # external inputs and synapses
    extS = PoissonGroup(N_X, rates=nu_ext)

    synSXSE = Synapses(extS, senE, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSXSE')
    synSXSE.connect(condition=condsame, p='epsX * (1 + alphaX)')
    synSXSE.connect(condition=conddiff, p='epsX * (1 - alphaX)')
    synSXSE.w = 'gXE/gleakE * (1 + randn()*0.5)'
    synSXSE.delay = dX

    synSXSI = Synapses(extS, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=paramsen, name='synSXSI')
    synSXSI.connect(p='epsX')
    synSXSI.w = 'gXI/gleakI * (1 + randn()*0.5)'
    synSXSI.delay = dX

    # variables to return
    groups = {'SE': senE, 'SI': senI, 'SX': extS}
    subgroups = {'SE1': senE1, 'SE2': senE2}
    synapses = {'synSESE': synSESE, 'synSESI': synSESI,
                'synSISE': synSISE, 'synSISI': synSISI,
                'synSXSI': synSXSI, 'synSXSE': synSXSE}

    return groups, synapses, subgroups


def mk_sencircuit_2c(task_info):
    """
    Creates balanced network representing sensory circuit with 2c model in the excitatory neurons.

    returns:
        groups, synapses, subgroups

    groups and synapses have to be added to the "Network" in order to run the simulation
    subgroups is used for establishing connections between the sensory and integration circuit;
    do not add subgroups to the "Network"


    psr following the published Brian1 code in DB model from Wimmer et al. 2015
    - brain2 code
    - two-compartmental model following Naud & Sprekeler 2018
    - implementation runs in cpp_standalone mode, compatible with SNEP
    """
    # -------------------------------------
    # Sensory circuit parameters
    # -------------------------------------
    # populations
    N_E = task_info['sen']['populations']['N_E']  # total number of E neurons
    sub = task_info['sen']['populations']['sub']  # fraction corresponding to subpops 1,2
    N_E1 = int(N_E * sub)  # size of exc subpops that responds to the stimuli
    N_I = task_info['sen']['populations']['N_I']  # total number of I neurons
    N_X = task_info['sen']['populations']['N_X']  # size of external pop

    # local recurrent connections
    eps = task_info['sen']['connectivity']['eps']  # connection probability for EE, EI, IE and II
    w_p = task_info['sen']['connectivity']['w_p']  # relative synaptic strength within pop E1 and E2
    w_m = 2 - w_p  # relative synaptic strength across pop E1 and E2
    gEEs = task_info['sen']['2c']['gEEs']  # weight of EE synapses, prev: 0.7589*nS
    gEI = task_info['sen']['connectivity']['gEI']  # weight of EI synapses, prev: 1.5179*nS
    gIEs = task_info['sen']['2c']['gIEs']  # weight of IE synapses, prev: 12.6491*nS
    gII = task_info['sen']['connectivity']['gII']  # weight of II synapses
    gmax = task_info['sen']['connectivity']['gmax']  # maximum synaptic weight
    dE = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of E synapses, (0.5:1.5)
    dI = '0.1*ms + rand()*0.8*ms'  # range of transmission delays of I synapses, (0.1:0.9)

    # external connections
    epsX = task_info['sen']['connectivity']['epsX']         # connection probability for ext synapses
    alphaX = task_info['sen']['connectivity']['alphaX']     # 0: global input, 1: local input
    gXEs = task_info['sen']['2c']['gXEs']                   # weight of ext to E synapses of soma
    gXI = task_info['sen']['connectivity']['gXI']           # weight of ext to I synapses
    dX = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of X synapses, (0.5:1.5)

    # neuron model
    CmI = task_info['sen']['neuron']['CmI']  # membrane capacitance of I neurons
    gleakI = task_info['sen']['neuron']['gleakI']  # leak conductance of I neurons
    Vl = task_info['sen']['neuron']['Vl']  # reversal potential
    Vt = task_info['sen']['neuron']['Vt']  # threshold potential
    Vr = task_info['sen']['neuron']['Vr']  # reset potential, only for inh
    tau_refI = task_info['sen']['neuron']['tau_refI']  # absolute refractory period of I neurons
    nu_ext = task_info['sen']['neuron']['nu_ext']  # rate of external Poisson neurons, prev: 12.5*Hz

    # soma
    taus = task_info['sen']['2c']['taus']  # timescale of membrane potential
    Cms = task_info['sen']['2c']['Cms']  # capacitance
    gleakEs = Cms/taus
    tauws = task_info['sen']['2c']['tauws']  # timescale of recovery ("slow") variable
    bws = task_info['sen']['2c']['bws']  # strength of spike-triggered facilitation (bws < 0)
    gCas = task_info['sen']['2c']['gCas']  # strenght of forward calcium spike propagation
    tau_refE = task_info['sen']['2c']['tau_refE']  # refractory period

    # dendrite
    taud = task_info['sen']['2c']['taud']
    Cmd = task_info['sen']['2c']['Cmd']
    gleakEd = Cmd/taud
    tauwd = task_info['sen']['2c']['tauwd']  # timescale of recovery ("slow") variable
    awd = task_info['sen']['2c']['awd']  # stregth of subthreshold facilitations (awd < 0)
    gCad = task_info['sen']['2c']['gCad']  # strength of local regenerative activity
    bpA = task_info['sen']['2c']['bpA']  # strength of backpropagation activity (c variable)
    k1 = task_info['sen']['2c']['k1']  # rectangular kernel for backpropagating activity
    k2 = task_info['sen']['2c']['k2']  # if t in [0.5, 2.0] we'll have backpropagating current
    muOUd = task_info['sen']['2c']['muOUd']    # OU parameters for background noise of dendrites
    sigmaOU = task_info['sen']['2c']['sigmaOU']
    tauOU = task_info['sen']['2c']['tauOU']

    # synapse models
    VrevE = task_info['sen']['synapse']['VrevE']  # reversal potential for E synapses
    VrevI = task_info['sen']['synapse']['VrevI']  # reversal potential for I synapses in inh
    tau_decay = task_info['sen']['synapse']['tau_decay']  # decay constant of AMPA, GABA
    tau_rise = task_info['sen']['synapse']['tau_rise']    # rise constant of AMPA, GABA  for inh
    VrevIsd = task_info['sen']['synapse']['VrevIsd'] # rev potential for I synapses in soma, dend
    #tau_decayE = task_info['sen']['synapse']['tau_decaysd']  # rise constants of AMPA conductances
    # in Wimmer are decayE = decayI = 5, Naud decayE = 1, decayI = 5 !
    # TODO: try both approaches and decide which one to use.

    param2c = {'gEE': gEEs, 'gEI': gEI, 'gIE': gIEs, 'gII': gII, 'gmax': gmax, 'gXEs': gXEs, 'gXI': gXI,
               'gleakEs': gleakEs, 'gleakEd': gleakEd, 'gleakI': gleakI, 'Cms': Cms, 'Cmd': Cmd, 'CmI': CmI,
               'Vl': Vl, 'Vt': Vt, 'Vr': Vr, 'VrevE': VrevE, 'VrevIsd': VrevIsd, 'VrevI': VrevI,
               'taus': taus, 'taud': taud, 'tau_refE': tau_refE, 'tau_refI': tau_refI,
               'tau_decay': tau_decay, 'tau_rise': tau_rise, 'tau_ws': tauws, 'tau_wd': tauwd,
               'bws': bws, 'awd': awd, 'gCas': gCas, 'gCad': gCad, 'bpA': bpA, 'k1': k1, 'k2': k2,
               'muOUd': muOUd, 'tauOU': tauOU, 'sigmaOU': sigmaOU,
               'w_p': w_p, 'w_m': w_m, 'sub': sub, 'eps': eps, 'epsX': epsX, 'alphaX': alphaX}

    # numerical integration method
    nummethod = task_info['simulation']['nummethod']

    # -------------------------------------
    # Set up model and connections
    # -------------------------------------
    # neuron equations
    eqss = '''
        dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevIsd) -(V-Vl))/tau + (gCas/(1+exp(-(V_d/mV + 38)/6)) + w_s + I)/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt  = -x_ea / tau_rise            : 1
        dg_i/dt  = (-g_i + x_i) / tau_decay     : 1
        dx_i/dt  = -x_i / tau_rise              : 1
        dw_s/dt = -w_s / tau_ws                 : amp
        tau = taus      : second
        Cm  = Cms       : farad
        I = Irec(t, i)  : amp
        V_d : volt (linked)
    '''

    eqsd = '''
        dV_d/dt = (-g_ea*(V_d-VrevE) -(V_d-Vl))/tau + (gCad/(1+exp(-(V_d/mV + 38)/6)) + w_d + K + Ibg)/Cm : volt
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt  = -x_ea / tau_rise            : 1
        dw_d/dt = (-w_d + awd*(V_d - Vl))/tau_wd : amp
        dIbg/dt = (muOUd - Ibg) / tauOU + (sigmaOU * xi) / sqrt(tauOU / 2)  : amp
        K = bpA * (((t-lastspike_soma) >= k1) * ((t-lastspike_soma) <= k2)) : amp
        tau = taud      : second
        Cm  = Cmd       : farad
        muOUd           : amp
        lastspike_soma  : second (linked)
        
    '''

    eqsI = '''
        dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevI) -(V-Vl))/tau  + I/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt = -x_ea / tau_rise             : 1
        dg_i/dt = (-g_i + x_i) / tau_decay      : 1
        dx_i/dt = -x_i / tau_rise               : 1
        tau = CmI/gleakI    : second
        Cm = CmI            : farad
        I : amp
    '''

    # neuron groups
    soma = NeuronGroup(N_E, model=eqss, method=nummethod, threshold='V>=Vt',
                       reset='''V = Vl
                                w_s += bws''',
                       refractory=tau_refE, namespace=param2c, name='soma')
    dend = NeuronGroup(N_E, model=eqsd, method=nummethod, namespace=param2c, name='dend')
    senI = NeuronGroup(N_I, model=eqsI, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refI, namespace=param2c, name='senI')

    # linked variables
    soma.V_d = linked_var(dend, 'V_d')
    dend.lastspike_soma = linked_var(soma, 'lastspike')

    # subgroups
    soma1 = soma[:N_E1]
    dend1 = dend[:N_E1]
    soma2 = soma[N_E1:]
    dend2 = dend[N_E1:]

    # synapses
    # weight according the different subgroups
    condsame = '(i<N_pre*sub and j<N_post*sub) or (i>=N_pre*sub and j>=N_post*sub)'
    conddiff = '(i<N_pre*sub and j>=N_post*sub) or (i>=N_pre*sub and j<N_post*sub)'

    # AMPA: exc --> exc
    synSESE = Synapses(soma, soma, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSESE')
    synSESE.connect(p='eps')
    synSESE.w[condsame] = 'w_p * gEE/gleakEs * (1 + randn()*0.5)'
    synSESE.w[conddiff] = 'w_m * gEE/gleakEs * (1 + randn()*0.5)'
    synSESE.delay = dE

    # AMPA: exc --> inh
    synSESI = Synapses(soma, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSESI')
    synSESI.connect(p='eps')
    synSESI.w = 'gEI/gleakI * (1 + randn()*0.5)'
    synSESI.delay = dE

    # GABA: inh --> exc
    synSISE = Synapses(senI, soma, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSISE')
    synSISE.connect(p='eps')
    synSISE.w = 'gIE/gleakEs * (1 + randn()*0.5)'
    synSISE.delay = dI

    # GABA: inh --> inh
    synSISI = Synapses(senI, senI, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSISI')
    synSISI.connect(p='eps')
    synSISI.w = 'gII/gleakI * (1 + randn()*0.5)'
    synSISI.delay = dI

    # external inputs and synapses
    extS = PoissonGroup(N_X, rates=nu_ext)

    synSXSEs = Synapses(extS, soma, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSXSEs')
    synSXSEs.connect(condition=condsame, p='epsX * (1 + alphaX)')
    synSXSEs.connect(condition=conddiff, p='epsX * (1 - alphaX)')
    synSXSEs.w = 'gXEs/gleakEs * (1 + randn()*0.5)'
    synSXSEs.delay = dX

    synSXSI = Synapses(extS, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSXSI')
    synSXSI.connect(p='epsX')
    synSXSI.w = 'gXI/gleakI * (1 + randn()*0.5)'
    synSXSI.delay = dX

    # variables to return
    groups = {'soma': soma, 'dend': dend, 'SI': senI, 'SX': extS}
    subgroups = {'soma1': soma1, 'soma2': soma2,
                 'dend1': dend1, 'dend2': dend2}
    synapses = {'synSESE': synSESE, 'synSESI': synSESI,
                'synSISE': synSISE, 'synSISI': synSISI,
                'synSXSI': synSXSI, 'synSXSEs': synSXSEs}

    return groups, synapses, subgroups


def mk_sencircuit_2cplastic(task_info):
    """
    Creates balanced network representing sensory circuit with 2c model in the excitatory neurons.

    returns:
        groups, synapses, subgroups

    groups and synapses have to be added to the "Network" in order to run the simulation
    subgroups is used for establishing connections between the sensory and integration circuit;
    do not add subgroups to the "Network"


    psr following the published Brian1 code in DB model from Wimmer et al. 2015
    - brain2 code
    - two-compartmental model following Naud & Sprekeler 2018
    - implementation runs in cpp_standalone mode, compatible with SNEP
    """
    from numpy import log as np_log

    # -------------------------------------
    # Sensory circuit parameters
    # -------------------------------------
    # populations
    N_E = task_info['sen']['populations']['N_E']  # total number of E neurons
    sub = task_info['sen']['populations']['sub']  # fraction corresponding to subpops 1,2
    N_E1 = int(N_E * sub)  # size of exc subpops that responds to the stimuli
    N_I = task_info['sen']['populations']['N_I']  # total number of I neurons
    N_X = task_info['sen']['populations']['N_X']  # size of external pop

    # local recurrent connections
    eps = task_info['sen']['connectivity']['eps']  # connection probability for EE, EI, IE and II
    w_p = task_info['sen']['connectivity']['w_p']  # relative synaptic strength within pop E1 and E2
    w_m = 2 - w_p  # relative synaptic strength across pop E1 and E2
    gEEs = task_info['sen']['2c']['gEEs']  # weight of EE synapses, prev: 0.7589*nS
    gEI = task_info['sen']['connectivity']['gEI']  # weight of EI synapses, prev: 1.5179*nS
    gIEs = task_info['sen']['2c']['gIEs']  # weight of IE synapses, prev: 12.6491*nS
    gII = task_info['sen']['connectivity']['gII']  # weight of II synapses
    gmax = task_info['sen']['connectivity']['gmax']  # maximum synaptic weight
    dE = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of E synapses, (0.5:1.5)
    dI = '0.1*ms + rand()*0.8*ms'  # range of transmission delays of I synapses, (0.1:0.9)

    # external connections
    epsX = task_info['sen']['connectivity']['epsX']         # connection probability for ext synapses
    alphaX = task_info['sen']['connectivity']['alphaX']     # 0: global input, 1: local input
    gXEs = task_info['sen']['2c']['gXEs']                   # weight of ext to E synapses of soma
    gXI = task_info['sen']['connectivity']['gXI']           # weight of ext to I synapses
    dX = '0.5*ms + rand()*1.0*ms'  # range of transmission delays of X synapses, (0.5:1.5)

    # neuron model
    CmI = task_info['sen']['neuron']['CmI']  # membrane capacitance of I neurons
    gleakI = task_info['sen']['neuron']['gleakI']  # leak conductance of I neurons
    Vl = task_info['sen']['neuron']['Vl']  # reversal potential
    Vt = task_info['sen']['neuron']['Vt']  # threshold potential
    Vr = task_info['sen']['neuron']['Vr']  # reset potential, only for inh
    tau_refI = task_info['sen']['neuron']['tau_refI']  # absolute refractory period of I neurons
    nu_ext = task_info['sen']['neuron']['nu_ext']  # rate of external Poisson neurons, prev: 12.5*Hz

    # soma
    taus = task_info['sen']['2c']['taus']  # timescale of membrane potential
    Cms = task_info['sen']['2c']['Cms']  # capacitance
    gleakEs = Cms/taus
    tauws = task_info['sen']['2c']['tauws']  # timescale of recovery ("slow") variable
    bws = task_info['sen']['2c']['bws']  # strength of spike-triggered facilitation (bws < 0)
    gCas = task_info['sen']['2c']['gCas']  # strenght of forward calcium spike propagation
    tau_refE = task_info['sen']['2c']['tau_refE']  # refractory period

    # dendrite
    taud = task_info['sen']['2c']['taud']
    Cmd = task_info['sen']['2c']['Cmd']
    gleakEd = Cmd/taud
    tauwd = task_info['sen']['2c']['tauwd']  # timescale of recovery ("slow") variable
    awd = task_info['sen']['2c']['awd']  # stregth of subthreshold facilitations (awd < 0)
    gCad = task_info['sen']['2c']['gCad']  # strength of local regenerative activity
    bpA = task_info['sen']['2c']['bpA']  # strength of backpropagation activity (c variable)
    k1 = task_info['sen']['2c']['k1']  # rectangular kernel for backpropagating activity
    k2 = task_info['sen']['2c']['k2']  # if t in [0.5, 2.0] we'll have backpropagating current
    muOUs = task_info['sen']['2c']['muOUs']    # OU parameters for background noise of soma
    #muOUd = task_info['sen']['2c']['muOUd']    # OU parameters for background noise of dendrites
    sigmaOU = task_info['sen']['2c']['sigmaOU']
    tauOU = task_info['sen']['2c']['tauOU']

    # synapse models
    VrevE = task_info['sen']['synapse']['VrevE']  # reversal potential for E synapses
    VrevI = task_info['sen']['synapse']['VrevI']  # reversal potential for I synapses in inh
    tau_decay = task_info['sen']['synapse']['tau_decay']  # decay constant of AMPA, GABA
    tau_rise = task_info['sen']['synapse']['tau_rise']    # rise constant of AMPA, GABA  for inh
    VrevIsd = task_info['sen']['synapse']['VrevIsd'] # rev potential for I synapses in soma, dend

    # plasticity in dendrites
    eta0 = task_info['sen']['2c']['eta0']
    tauB = task_info['sen']['2c']['tauB']
    targetB = task_info['targetB']
    B0 = tauB*targetB
    tau_update = task_info['sen']['2c']['tau_update']
    eta = eta0 * tau_update / tauB
    validburst = task_info['sen']['2c']['validburst']      # if tau_burst = 4*ms, at 16 ms stopburst = 0.0183
    min_burst_stop = task_info['sen']['2c']['min_burst_stop']   # thus, we will set it at critrefburst = 0.02
    tau_burst = -validburst / np_log(min_burst_stop) * second

    param2c = {'gEE': gEEs, 'gEI': gEI, 'gIE': gIEs, 'gII': gII, 'gmax': gmax, 'gXEs': gXEs, 'gXI': gXI,
               'gleakEs': gleakEs, 'gleakEd': gleakEd, 'gleakI': gleakI, 'Cms': Cms, 'Cmd': Cmd, 'CmI': CmI,
               'Vl': Vl, 'Vt': Vt, 'Vr': Vr, 'VrevE': VrevE, 'VrevIsd': VrevIsd, 'VrevI': VrevI,
               'taus': taus, 'taud': taud, 'tau_refE': tau_refE, 'tau_refI': tau_refI,
               'tau_decay': tau_decay, 'tau_rise': tau_rise, 'tau_ws': tauws, 'tau_wd': tauwd,
               'bws': bws, 'awd': awd, 'gCas': gCas, 'gCad': gCad, 'bpA': bpA, 'k1': k1, 'k2': k2,
               'muOUs': muOUs, 'tauOU': tauOU, 'sigmaOU': sigmaOU,
               'w_p': w_p, 'w_m': w_m, 'sub': sub, 'eps': eps, 'epsX': epsX, 'alphaX': alphaX,
               'eta': eta, 'B0': B0, 'tauB': tauB, 'tau_burst': tau_burst, 'min_burst_stop': min_burst_stop}

    # numerical integration method
    nummethod = task_info['simulation']['nummethod']

    # -------------------------------------
    # Set up model and connections
    # -------------------------------------
    # neuron equations
    eqss = '''
        dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevIsd) -(V-Vl))/tau + (gCas/(1+exp(-(V_d/mV + 38)/6)) + w_s + I)/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt  = -x_ea / tau_rise            : 1
        dg_i/dt  = (-g_i + x_i) / tau_decay     : 1
        dx_i/dt  = -x_i / tau_rise              : 1
        dw_s/dt = -w_s / tau_ws                 : amp
        tau = taus      : second
        Cm  = Cms       : farad
        I = Irec(t, i)  : amp
        V_d             : volt (linked)
        B               : 1 (linked)
        burst_start     : 1 (linked)
        burst_stop      : 1 (linked)
        muOUd           : amp (linked)
    '''
        # dIbg/dt = (muOUs - Ibg) / tauOU + (sigmaOU * xi) / sqrt(tauOU / 2)  : amp --> they have I_Ext

    eqsd = '''
        dV_d/dt = (-g_ea*(V_d-VrevE) -(V_d-Vl))/tau + (gCad/(1+exp(-(V_d/mV + 38)/6)) + w_d + K + Ibg)/Cm : volt
        dg_ea/dt = (-g_ea + x_ea) / tau_decay          : 1
        dx_ea/dt  = -x_ea / tau_rise                   : 1
        dw_d/dt = (-w_d + awd*(V_d - Vl))/tau_wd       : amp
        dIbg/dt = (muOUd - Ibg) / tauOU + (sigmaOU * xi) / sqrt(tauOU / 2)  : amp
        K = bpA * (((t-lastspike_soma) >= k1) * ((t-lastspike_soma) <= k2))   : amp
        dB/dt = -B / tauB                              : 1
        dburst_start/dt = -burst_start / tau_burst     : 1 (unless refractory)
        dburst_stop/dt = -burst_stop / tau_burst       : 1
        tau = taud      : second
        Cm  = Cmd       : farad
        muOUd           : amp
        lastspike_soma  : second (linked)
    '''

    eqsI = '''
        dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevI) -(V-Vl))/tau  + I/Cm : volt (unless refractory)
        dg_ea/dt = (-g_ea + x_ea) / tau_decay   : 1
        dx_ea/dt = -x_ea / tau_rise             : 1
        dg_i/dt = (-g_i + x_i) / tau_decay      : 1
        dx_i/dt = -x_i / tau_rise               : 1
        tau = CmI/gleakI    : second
        Cm = CmI            : farad
        I : amp
    '''

    # neuron groups
    soma = NeuronGroup(N_E, model=eqss, method=nummethod, threshold='V>=Vt',
                       reset='''V = Vl
                                w_s += bws
                                burst_start += 1
                                burst_stop = 1''',
                       refractory=tau_refE, namespace=param2c, name='soma')
    dend = NeuronGroup(N_E, model=eqsd, method=nummethod, threshold='burst_start > 1 + min_burst_stop',
                       reset='''B += 1
                                burst_start = 0''',
                       refractory='burst_stop >= min_burst_stop',
                       namespace=param2c, name='dend')
    senI = NeuronGroup(N_I, model=eqsI, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refI, namespace=param2c, name='senI')

    # linked variables
    soma.V_d = linked_var(dend, 'V_d')
    soma.B = linked_var(dend, 'B')
    soma.burst_start = linked_var(dend, 'burst_start')
    soma.burst_stop = linked_var(dend, 'burst_stop')
    soma.muOUd = linked_var(dend, 'muOUd')
    dend.lastspike_soma = linked_var(soma, 'lastspike')

    # subgroups
    soma1 = soma[:N_E1]
    dend1 = dend[:N_E1]
    soma2 = soma[N_E1:]
    dend2 = dend[N_E1:]

    # update muOUd with plasticity rule
    dend1.muOUd = '-50*pA - rand()*100*pA'   # random initialisation of weights -50:-150 pA
    dend1.run_regularly('muOUd = clip(muOUd - eta * (B - B0), -100*amp, 0)', dt=tau_update)
    # TODO: check the target vs burst rate convergence
    # TODO: try tau_B = 50,000 sec, but mayb\e this is not the case because you do reach the target

    # synapses
    # weight according the different subgroups
    condsame = '(i<N_pre*sub and j<N_post*sub) or (i>=N_pre*sub and j>=N_post*sub)'
    conddiff = '(i<N_pre*sub and j>=N_post*sub) or (i>=N_pre*sub and j<N_post*sub)'

    # AMPA: exc --> exc
    synSESE = Synapses(soma, soma, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSESE')
    synSESE.connect(p='eps')
    synSESE.w[condsame] = 'w_p * gEE/gleakEs * (1 + randn()*0.5)'
    synSESE.w[conddiff] = 'w_m * gEE/gleakEs * (1 + randn()*0.5)'
    synSESE.delay = dE

    # AMPA: exc --> inh
    synSESI = Synapses(soma, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSESI')
    synSESI.connect(p='eps')
    synSESI.w = 'gEI/gleakI * (1 + randn()*0.5)'
    synSESI.delay = dE

    # GABA: inh --> exc
    synSISE = Synapses(senI, soma, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSISE')
    synSISE.connect(p='eps')
    synSISE.w = 'gIE/gleakEs * (1 + randn()*0.5)'
    synSISE.delay = dI

    # GABA: inh --> inh
    synSISI = Synapses(senI, senI, model='w : 1', method=nummethod,
                       on_pre='''x_i += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSISI')
    synSISI.connect(p='eps')
    synSISI.w = 'gII/gleakI * (1 + randn()*0.5)'
    synSISI.delay = dI

    # external inputs and synapses
    extS = PoissonGroup(N_X, rates=nu_ext)

    synSXSEs = Synapses(extS, soma, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSXSEs')
    synSXSEs.connect(condition=condsame, p='epsX * (1 + alphaX)')
    synSXSEs.connect(condition=conddiff, p='epsX * (1 - alphaX)')
    synSXSEs.w = 'gXEs/gleakEs * (1 + randn()*0.5)'
    synSXSEs.delay = dX

    synSXSI = Synapses(extS, senI, model='w : 1', method=nummethod,
                       on_pre='''x_ea += w
                                 w = clip(w, 0, gmax)''',
                       namespace=param2c, name='synSXSI')
    synSXSI.connect(p='epsX')
    synSXSI.w = 'gXI/gleakI * (1 + randn()*0.5)'
    synSXSI.delay = dX

    # external inputs to dendrites mimicking decision circuit
    subDE = 240
    rateDE = 40*Hz   # TODO: lower rate --> 40 or 30
    d = 1*ms
    wDS = 0.06
    XDE = PoissonGroup(subDE, rates=rateDE)

    synXDEdend = Synapses(XDE, dend1, model='w : 1', method=nummethod, delay=d,
                          on_pre='x_ea += w',
                          namespace=param2c, name='synXDEdend')
    synXDEdend.connect(p=0.2)
    synXDEdend.w = 'wDS'

    # variables to return
    groups = {'soma': soma, 'dend': dend, 'SI': senI, 'SX': extS, 'XDE': XDE}
    subgroups = {'soma1': soma1, 'soma2': soma2,
                 'dend1': dend1, 'dend2': dend2}
    synapses = {'synSESE': synSESE, 'synSESI': synSESI,
                'synSISE': synSISE, 'synSISI': synSISI,
                'synSXSI': synSXSI, 'synSXSEs': synSXSEs,
                'synXDEdend': synXDEdend}

    return groups, synapses, subgroups
