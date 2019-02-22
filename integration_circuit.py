from brian2 import PoissonInput, linked_var
from brian2.groups import NeuronGroup
from brian2.synapses import Synapses


def mk_intcircuit(task_info):
    """
    Creates the 'winner-takes-all' network described in Wang 2002.

    returns:
        groups, synapses, update_nmda, subgroups

    groups, synapses and update_nmda have to be added to the "Network" in order to run the simulation
    subgroups is used for establishing connections between the sensory and integration circuit;
    do not add subgroups to the "Network"

    psr following the published Brian1 code in DB model from Wimmer et al. 2015
    - brian2 code
    - PoissonInput for external connections
    - no more network_operations
    - implementation runs in cpp_standalone mode, compatible with SNEP
    """
    # -------------------------------------
    # Decision circuit parameters
    # -------------------------------------
    # populations
    N_E = task_info['dec']['populations']['N_E']  # number of exc neurons (1600)
    N_I = task_info['dec']['populations']['N_I']  # number of inh neurons (400)
    sub = task_info['dec']['populations']['sub']  # fraction of stim-selective exc neurons
    N_D1 = int(N_E * sub)  # size of exc pop D1
    N_D2 = N_D1  # size of exc pop D2
    N_D3 = int(N_E * (1 - 2 * sub))  # size of exc pop D3, the rest

    # local recurrent connections
    w_p = task_info['dec']['connectivity']['w_p']  # relative synaptic strength of synapses within pop D1 and D2
    w_m = 1 - sub * (w_p - 1) / (1 - sub)  # relative synaptic strength of synapses across pop D1 and D2
    gEEa = task_info['dec']['connectivity']['gEEa']  # AMPA weight of EE synapses
    gEEn = task_info['dec']['connectivity']['gEEn']  # NMDA weight of EE synapses
    gEIa = task_info['dec']['connectivity']['gEIa']  # AMPA weight of EI synapses
    gEIn = task_info['dec']['connectivity']['gEIn']  # NMDA weight of EI synapses
    gIE = task_info['dec']['connectivity']['gIE']  # GABA weight of IE synapses, vs 1.3*nS from before
    gII = task_info['dec']['connectivity']['gII']  # GABA weight of II synapses
    d = task_info['dec']['connectivity']['delay']  # transmission delays of E synapses

    # external connections
    gXE = task_info['dec']['connectivity']['gXE']  # weight of XE (ext to exc) synapses
    gXI = task_info['dec']['connectivity']['gXI']  # weight of XI (ext to inh) synapses

    # neuron models
    CmE = task_info['dec']['neuron']['CmE']  # membrane capacitance of E neurons
    CmI = task_info['dec']['neuron']['CmI']  # membrane capacitance of I neurons
    gleakE = task_info['dec']['neuron']['gleakE']  # leak conductance of E neurons
    gleakI = task_info['dec']['neuron']['gleakI']  # leak conductance of I neurons
    Vl = task_info['dec']['neuron']['Vl']  # resting potential
    Vt = task_info['dec']['neuron']['Vt']  # spiking threshold
    Vr = task_info['dec']['neuron']['Vr']  # reset potential
    tau_refE = task_info['dec']['neuron']['tau_refE']  # absolute refractory period of E neurons
    tau_refI = task_info['dec']['neuron']['tau_refI']  # absolute refractory period of I neurons
    nu_ext = task_info['dec']['neuron']['nu_ext']  # firing rate of ext Poisson input to D1 and D2
    nu_ext1 = task_info['dec']['neuron']['nu_ext1']  # firing rate of ext Poisson input to D3 and DI

    # synapse models
    VrevE = task_info['dec']['synapse']['VrevE']  # reversal potential for E synapses
    VrevI = task_info['dec']['synapse']['VrevI']  # reversal potential for I synapses
    tau_ampa = task_info['dec']['synapse']['tau_ampa']  # decay constant of AMPA conductances
    tau_gaba = task_info['dec']['synapse']['tau_gaba']  # decay constant of GABA conductances
    tau_nmda_d = task_info['dec']['synapse']['tau_nmda_d']  # decay constant of NMDA conductances
    tau_nmda_r = task_info['dec']['synapse']['tau_nmda_r']  # rise constant of NMDA conductances
    alpha_nmda = task_info['dec']['synapse']['alpha_nmda']  # saturation constant of NMDA conductances

    # namespace with params
    paramint = {'w_p': w_p, 'w_m': w_m, 'gEEa': gEEa, 'gEEn': gEEn, 'gEIa': gEIa, 'gEIn': gEIn,
                'gIE': gIE, 'gII': gII, 'gXE': gXE, 'gXI': gXI, 'gleakE': gleakE, 'gleakI': gleakI,
                'Vl': Vl, 'Vt': Vt, 'Vr': Vr, 'VrevE': VrevE, 'VrevI': VrevI, 'tau_ampa': tau_ampa,
                'tau_gaba': tau_gaba, 'tau_nmda_d': tau_nmda_d, 'tau_nmda_r': tau_nmda_r, 'alpha_nmda': alpha_nmda,
                'sub': sub, 'CmE': CmE, 'CmI': CmI}

    # numerical integration method
    nummethod = task_info['simulation']['nummethod']

    # -------------------------------------
    # Set up the model and connections
    # -------------------------------------
    # neuron equations
    eqsE = '''
            dV/dt = (-g_ea*(V-VrevE) - g_ent*(V-VrevE)/(1+exp(-V/mV*0.062)/3.57) - g_i*(V-VrevI) - (V-Vl)) / tau : volt (unless refractory)
            dg_ea/dt = -g_ea / tau_ampa     : 1
            dg_i/dt  = -g_i / tau_gaba      : 1
            dg_en/dt = -g_en / tau_nmda_d + alpha_nmda * x_en *(1-g_en) : 1
            dx_en/dt = -x_en / tau_nmda_r   : 1
            g_ent               : 1
            tau = CmE/gleakE    : second
            label : integer (constant)
            '''

    eqsI = '''
            dV/dt = (-g_ea*(V-VrevE) - g_entI*(V-VrevE)/(1+exp(-V/mV*0.062)/3.57) - g_i*(V-VrevI) - (V-Vl)) / tau : volt (unless refractory)
            dg_ea/dt = -g_ea/tau_ampa       : 1
            dg_i/dt  = -g_i/tau_gaba        : 1
            g_entI = w_nmda * g_ent         : 1
            g_ent               : 1 (linked)
            w_nmda              : 1
            tau = CmI/gleakI    : second
            '''

    # setup of integration circuit
    decE = NeuronGroup(N_E, model=eqsE, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refE, namespace=paramint, name='decE')
    decE1 = decE[:N_D1]
    decE2 = decE[N_D1:N_D1 + N_D2]
    decE3 = decE[-N_D3:]
    decE1.label = 1
    decE2.label = 2
    decE3.label = 3

    decI = NeuronGroup(N_I, model=eqsI, method=nummethod, threshold='V>=Vt', reset='V=Vr',
                       refractory=tau_refI, namespace=paramint, name='decI')

    # weight according the different subgroups
    condsame = '(label_pre == label_post and label_pre != 3)'
    conddiff = '(label_pre != label_post and label_pre != 3) or (label_pre == 3 and label_post != 3)'
    condrest = '(label_post == 3)'

    # NMDA: exc --> exc
    eqsNMDA = '''
            g_ent_post = w_nmda * g_en_pre      : 1 (summed)
            w_nmda  : 1 (constant)
            w       : 1 (constant)
            '''

    synDEDEn = Synapses(decE, decE, model=eqsNMDA, method=nummethod, on_pre='x_en += w', delay=d,
                        namespace=paramint, name='synDEDEn')
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
    synDEDEa = Synapses(decE, decE, model='w : 1', method=nummethod,
                        on_pre='g_ea += w', delay=d,
                        namespace=paramint, name='synDEDEa')
    synDEDEa.connect()
    synDEDEa.w[condsame] = 'w_p * gEEa/gleakE'
    synDEDEa.w[conddiff] = 'w_m * gEEa/gleakE'
    synDEDEa.w[condrest] = 'gEEa/gleakE'

    # AMPA: exc --> inh
    synDEDIa = Synapses(decE, decI, model='w : 1', method=nummethod,
                        on_pre='g_ea += w', delay=d,
                        namespace=paramint, name='synDEDIa')
    synDEDIa.connect()
    synDEDIa.w = 'gEIa/gleakI'

    # GABA: inh --> exc
    synDIDE = Synapses(decI, decE, model='w : 1', method=nummethod,
                       on_pre='g_i += w', delay=d,
                       namespace=paramint, name='synDIDE')
    synDIDE.connect()
    synDIDE.w = 'gIE/gleakE'

    # GABA: inh --> inh
    synDIDI = Synapses(decI, decI, model='w : 1', method=nummethod,
                       on_pre='g_i += w', delay=d,
                       namespace=paramint, name='synDIDI')
    synDIDI.connect()
    synDIDI.w = 'gII/gleakI'

    # external inputs and connections
    extE = PoissonInput(decE[:N_D1 + N_D2], 'g_ea', N=1, rate=nu_ext1, weight='gXE/gleakE')
    extE3 = PoissonInput(decE3, 'g_ea', N=1, rate=nu_ext, weight='gXE/gleakE')
    extI = PoissonInput(decI, 'g_ea', N=1, rate=nu_ext, weight='gXI/gleakI')

    # variables to return
    groups = {'DE': decE, 'DI': decI, 'DX': extE, 'DX3': extE3, 'DXI': extI}
    subgroups = {'DE1': decE1, 'DE2': decE2, 'DE3': decE3}
    synapses = {'synDEDEn': synDEDEn,
                'synDEDEa': synDEDEa, 'synDEDIa': synDEDIa,
                'synDIDE': synDIDE, 'synDIDI': synDIDI}  # 'synDEDIn': synDEDIn,

    return groups, synapses, subgroups
