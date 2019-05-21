# all functions return a dictionary of parameters specific to each case.
from brian2.units import Hz, ms, nS, pF, mV, pA, second
from helper_funcs import adjust_variable


def get_dec_params(task_info):
    """Parameters for decision circuit."""
    # local recurrent connections
    sub = task_info['dec']['sub']       # fraction of stim-selective exc neurons
    w_p = 1.6               # relative synaptic strength within pop D1 and D2
    w_m = 1 - sub*(w_p - 1) / (1 - sub)     # " " across pop D1 and D2
    gEEa = 0.05*nS          # AMPA weight of EE synapses
    gEEn = 0.165*nS         # NMDA weight of EE synapses
    gEIa = 0.04*nS          # AMPA weight of EI synapses
    gEIn = 0.13*nS          # NMDA weight of EI synapses
    gIE = 1.3*nS            # GABA weight of IE synapses, vs 1.33*nS from before
    gII = 1*nS              # GABA weight of II synapses
    d = 0.5*ms              # transmission delays of E synapses

    # external connections
    gXE = 2.1*nS            # weight of XE synapses
    gXI = 1.62*nS           # weight of XI synapses
    nu_ext_12 = 2392*Hz     # firing rate of ext Poisson input to D1 and D2
    nu_ext_3I = 2400*Hz     # firing rate of ext Poisson input to D3 and DI

    # neuron models
    CmE = 500*pF            # membrane capacitance of E neurons
    CmI = 200*pF            # membrane capacitance of I neurons
    gleakE = 25*nS          # leak conductance of E neurons
    gleakI = 20*nS          # leak conductance of I neurons
    Vl = -70*mV             # resting potential
    Vt = -50*mV             # spiking threshold
    Vr = -55*mV             # reset potential
    tau_refE = 2*ms         # absolute refractory period of E neurons
    tau_refI = 1*ms         # absolute refractory period of I neurons

    # synapse models
    VrevE = 0*mV            # reversal potential for E synapses
    VrevI = -70*mV          # reversal potential for I synapses
    tau_ampa_d = 2*ms       # decay constant of AMPA conductance
    tau_gaba_d = 5*ms       # decay constant of GABA conductance
    tau_nmda_d = 100*ms     # decay constant of NMDA conductance
    tau_nmda_r = 2*ms       # rise constant of NMDA conductance
    alpha_nmda = 500*Hz     # saturation constant of NMDA conductance

    # define namespace
    paramint = {'w_p': w_p, 'w_m': w_m, 'gEEa': gEEa, 'gEEn': gEEn, 'gEIa': gEIa, 'gEIn': gEIn, 'gIE': gIE, 'gII': gII,
                'gXE': gXE, 'gXI': gXI, 'gleakE': gleakE, 'gleakI': gleakI, 'Vl': Vl, 'Vt': Vt, 'Vr': Vr,
                'VrevE': VrevE, 'VrevI': VrevI, 'tau_refE': tau_refE, 'tau_refI': tau_refI, 'tau_ampa_d': tau_ampa_d,
                'tau_gaba_d': tau_gaba_d, 'tau_nmda_d': tau_nmda_d, 'tau_nmda_r': tau_nmda_r, 'alpha_nmda': alpha_nmda,
                'CmE': CmE, 'CmI': CmI, 'nu_ext_12': nu_ext_12, 'nu_ext_3I': nu_ext_3I, 'd': d}

    return paramint


def get_sen_params(task_info):
    """Parameters for sensory circuit."""
    # local recurrent connections
    eps = 0.2               # connection probability for EE, EI, IE and II
    w_p = 1.3               # synaptic strength of synapses within pop E1 and E2, w_m = 2 - w_p
    w_m = 2 - w_p           # relative synaptic strength across pop E1 and E2
    gEE = 0.7589*nS         # weight of EE synapses, prev: 0.7589*nS
    gEI = 1.5179*nS         # weight of EI synapses, prev: 1.5179*nS
    gIE = 12.6491*nS        # weight of IE synapses, prev: 12.6491*nS
    gII = gIE               # weight of II synapses
    gmax = 100              # maximum synaptic weight
    dE = '0.5*ms + rand()*1.0*ms'   # range of transmission delays of E synapses, (0.5:1.5)
    dI = '0.1*ms + rand()*0.8*ms'   # range of transmission delays of I synapses, (0.1:0.9)

    # external connections
    epsX = 0.32             # connection probability for ext synapses
    alphaX = 0              # 0: global input, 1: local input
    gXE = 1.7076*nS         # weight of XE synapses
    gXI = gXE               # weight of XI synapses
    nu_ext = 12.5*Hz        # firing rate of XPoisson neurons, prev: 12.5*Hz
    dX = '0.5*ms + rand()*1.0*ms'   # range of transmission delays of X synapses, (0.5:1.5)

    # neuron models
    CmE = 250*pF            # membrane capacitance of E neurons
    CmI = CmE               # membrane capacitance of I neurons
    gleakE = 16.7*nS        # leak conductance of E neurons
    gleakI = gleakE         # leak conductance of I neurons
    Vl = -70*mV             # resting potential
    Vt = -50*mV             # spiking threshold
    Vr = -60*mV             # reset potential
    tau_refE = 2*ms         # absolute refractory period of E neurons
    tau_refI = 1*ms         # absolute refractory period of I neurons

    # synapse models
    VrevE = 0*mV            # reversal potential for E synapses
    VrevI = -80*mV          # reversal potential for I synapses
    tau_d = 5*ms            # decay constants of AMPA and GABA conductance
    tau_r = 1*ms            # rise constants of AMPA and GABA conductance

    # define namespace
    paramsen = {'gEE': gEE, 'gEI': gEI, 'gIE': gIE, 'gII': gII, 'gmax': gmax, 'gXE': gXE, 'gXI': gXI, 'gleakE': gleakE,
                'gleakI': gleakI, 'CmE': CmE, 'CmI': CmI, 'Vl': Vl, 'Vt': Vt, 'Vr': Vr, 'VrevE': VrevE, 'VrevI': VrevI,
                'tau_d': tau_d, 'tau_r': tau_r, 'tau_refE': tau_refE, 'tau_refI': tau_refI, 'w_p': w_p, 'w_m': w_m,
                'eps': eps, 'epsX': epsX, 'alphaX': alphaX, 'nu_ext': nu_ext, 'dX': dX, 'dE': dE, 'dI': dI}

    return paramsen


def get_2c_params(task_info):
    """Parameters for two compartmental model of excitatory neurons within the sensory circuit."""
    # soma
    Cms = 370*pF            # capacitance
    taus = 14.97*ms         # timescale of membrane potential, original 16*ms
    gleakEs = Cms/taus
    tauws = 100*ms          # timescale of recovery ("slow") variable
    bws = -200*pA           # strength of spike-triggered facilitation (bws < 0)
    gCas = 1300*pA          # strength of forward calcium spike propagation
    tau_refE = 3*ms         # refractory period is 1*ms longer

    # dendrite
    Cmd = 170*pF
    taud = 7*ms
    gleakEd = Cmd/taud
    tauwd = 30*ms           # timescale of recovery ("slow") variable
    awd = -13*nS            # strength of sub-threshold facilitation (awd < 0)
    gCad = 1200*pA          # strength of local regenerative activity
    bpA = 2600*pA           # strength of back-propagation activity (c variable)
    k1 = 0.5*ms             # rectangular kernel for back-propagating activity
    k2 = 2.5*ms             # if t in [0.5, 2.5] we'll have back-propagating current

    # external/background noise
    muOUs = 70*pA           # drift of OU for soma
    sigmaOU = 450*pA        # diffusion of OU process
    # sigmaOU = 0 * pA  # diffusion of OU process
    tauOU = 2*ms            # timescale of OU process

    # synapse models
    VrevIsd = -70*mV        # rev potential for I synapses in soma, dend

    # previous sensory namespace
    param_wimmer = get_sen_params(task_info)
    param_naud = {'gleakE': gleakEs, 'gleakEd': gleakEd, 'Cms': Cms, 'Cmd': Cmd, 'VrevIsd': VrevIsd,
                  'taus': taus, 'taud': taud, 'tau_refE': tau_refE, 'tau_ws': tauws, 'tau_wd': tauwd,
                  'bws': bws, 'awd': awd, 'gCas': gCas, 'gCad': gCad, 'bpA': bpA, 'k1': k1, 'k2': k2,
                  'muOUs': muOUs, 'tauOU': tauOU, 'sigmaOU': sigmaOU}

    # merge dictionaries, order matters because previous variables get updated
    paramsen = {**param_wimmer, **param_naud}

    # re-adjust synaptic conductance
    for gx in ['gEE', 'gIE', 'gXE']:
        paramsen[gx] = adjust_variable(param_wimmer[gx], param_wimmer['gleakE'], gleakEs)
    # paramsen['gXE'] = 2.45*nS   # scaled: 2.527, lower_bound: 2.17. prev 2.4602 --> 550 pA

    return paramsen


def get_stim_params(task_info):
    """Parameters for creating an OU process as stimulus."""
    try:
        c = task_info['c']      # stim coherence (between 0 and 1)
    except KeyError:
        c = 0
    I0 = 80*pA                  # mean input current for zero-coherence stim. prev: 80*pA
    I0_wimmer = I0
    mu1 = 0.25                  # av. additional input current to pop1 at highest coherence
    mu2 = -0.25                 # av. additional input current to pop2 at highest coherence
    sigma = 1                   # amplitude of temporal modulations of stim
    sigma_stim = 0.212*sigma    # std of modulations of stim inputs
    sigma_ind = 0.212*sigma     # std of modulations in individual inputs
    tau_stim = 20*ms            # correlation time constant of OU process
    stim_dt = task_info['sim']['stim_dt']  # integration step of stimulus

    # adjust external current
    if task_info['sim']['2c_model']:
        paramsen = get_2c_params(task_info)
        I0 = 100*pA
        # I0 = adjust_variable(I0, paramsen['CmE'], paramsen['Cms'])  # 160*pA
        # I0_wimmer = I0  # makes sure the variance of the noise is also scaled!

    paramstim = {'c': c, 'I0': I0, 'I0_wimmer': I0_wimmer, 'mu1': mu1, 'mu2': mu2,  'tau_stim': tau_stim,
                 'stim_dt': stim_dt, 'sigma_stim': sigma_stim, 'sigma_ind': sigma_ind}

    return paramstim


def get_fffb_params(task_info):
    """Parameters for creating feedforward and feedback synapses for the hierachical network."""
    eps = 0.2                   # connection probability
    d = 1 * ms                  # transmission delays of E synapses
    w_ff = 0.0036               # weight of ff synapses, 0.09 nS when scaled by gleakE of dec_circuit
    w_fb = 0.004                # weight of fb synapses, 0.0668 nS when scaled by gleakE of sen_circuit_wimmer
    b_fb = task_info['bfb']     # feedback strength, between 0 and 6

    paramfffb = {'eps': eps, 'd': d, 'w_ff': w_ff, 'w_fb': w_fb*b_fb}

    return paramfffb


def get_plasticity_params(task_info):
    """Parameters for setting inhibitory plasticity rule on dendrites of sensory circuit."""
    from numpy import log as np_log
    tauB = task_info['plastic']['tauB']
    tau_update = task_info['plastic']['tau_update']
    eta0 = task_info['plastic']['eta0']
    valid_burst = task_info['sim']['valid_burst']
    min_burst_stop = task_info['plastic']['min_burst_stop']
    targetB = task_info['targetB']
    B0 = tauB * targetB
    eta = eta0 * tau_update / tauB
    tau_burst = -valid_burst / np_log(min_burst_stop) * second

    # twoComp sensory namespace
    param_naud = get_2c_params(task_info)
    param_plastic = {'tauB': tauB, 'tau_update': tau_update, 'tau_burst': tau_burst,
                     'valid_burst': valid_burst, 'min_burst_stop': min_burst_stop,
                     'B0': B0, 'eta': eta}
    paramplastic = {**param_naud, **param_plastic}

    return paramplastic
