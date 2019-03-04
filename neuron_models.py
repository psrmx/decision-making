# Explicit equations to model neurons in hierarchical network

# Sensory neurons
eqs_wimmer_exc = '''
    dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevI) -(V-Vl))/tau  + I/Cm : volt (unless refractory)
    dg_ea/dt = (-g_ea + x_ea) / tau_d       : 1
    dx_ea/dt = -x_ea / tau_r                : 1
    dg_i/dt = (-g_i + x_i) / tau_d          : 1
    dx_i/dt = -x_i / tau_r                  : 1
    tau = CmE/gleakE    : second
    Cm = CmE            : farad
    I = Irec(t, i)      : amp
'''

eqs_wimmer_inh = '''
    dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevI) -(V-Vl))/tau : volt (unless refractory)
    dg_ea/dt = (-g_ea + x_ea) / tau_d       : 1
    dx_ea/dt = -x_ea / tau_r                : 1
    dg_i/dt = (-g_i + x_i) / tau_d          : 1
    dx_i/dt = -x_i / tau_r                  : 1
    tau = CmI/gleakI    : second
    Cm = CmI            : farad
'''

eqs_naud_soma = '''
    dV/dt = (-g_ea*(V-VrevE) -g_i*(V-VrevIsd) -(V-Vl))/tau + (gCas/(1+exp(-(V_d/mV + 38)/6)) + w_s + I)/Cm : volt (unless refractory)
    dg_ea/dt = (-g_ea + x_ea) / tau_ampa_d  : 1
    dx_ea/dt = -x_ea / tau_r                : 1
    dg_i/dt = (-g_i + x_i) / tau_gaba_d     : 1
    dx_i/dt = -x_i / tau_r                  : 1
    dw_s/dt = -w_s / tau_ws                 : amp
    tau = taus          : second
    Cm = Cms            : farad
    I = Irec(t, i)      : amp
    V_d                 : volt (linked)
'''

eqs_naud_dend = '''
    dV_d/dt = (-g_ea*(V_d-VrevE) -(V_d-Vl))/tau + (gCad/(1+exp(-(V_d/mV + 38)/6)) + w_d + K + Ibg)/Cm : volt
    dg_ea/dt = (-g_ea + x_ea) / tau_ampa_d  : 1
    dx_ea/dt = -x_ea / tau_r                : 1
    dw_d/dt = (-w_d + awd * (V_d - Vl)) / tau_wd                        : amp
    dIbg/dt = (muOUd - Ibg) / tauOU + (sigmaOU * xi) / sqrt(tauOU / 2)  : amp
    K = bpA * (((t-lastspike_soma) >= k1) * ((t-lastspike_soma) <= k2)) : amp
    tau = taud          : second
    Cm = Cmd            : farad
    muOUd               : amp
    lastspike_soma      : second (linked)
'''

eqs_plasticity = '''
    dB/dt = -B / tauB                               : 1
    dburst_start/dt = -burst_start / tau_burst      : 1 (unless refractory)
    dburst_stop/dt = -burst_stop / tau_burst        : 1
'''

# Decision neurons
eqs_wang_exc = '''
    dV/dt = (-g_ea*(V-VrevE) - g_ent*(V-VrevE)/(1+exp(-V/mV*0.062)/3.57) - g_i*(V-VrevI) - (V-Vl)) / tau : volt (unless refractory)
    dg_ea/dt = -g_ea / tau_ampa_d           : 1
    dg_i/dt = -g_i / tau_gaba_d             : 1
    dg_en/dt = -g_en / tau_nmda_d + alpha_nmda * x_en * (1 - g_en)      : 1
    dx_en/dt = -x_en / tau_nmda_r           : 1
    g_ent               : 1
    tau = CmE/gleakE    : second
    label               : integer (constant)
'''

eqs_wang_inh = '''
    dV/dt = (-g_ea*(V-VrevE) - g_entI*(V-VrevE)/(1+exp(-V/mV*0.062)/3.57) - g_i*(V-VrevI) - (V-Vl)) / tau : volt (unless refractory)
    dg_ea/dt = -g_ea / tau_ampa_d           : 1
    dg_i/dt = -g_i / tau_gaba_d             : 1
    g_entI = w_nmda * g_ent                 : 1
    g_ent               : 1 (linked)
    w_nmda              : 1
    tau = CmI/gleakI    : second
'''

eqs_NMDA = '''
        g_ent_post = w_nmda * g_en_pre      : 1 (summed)
        w_nmda  : 1 (constant)
        w       : 1 (constant)
        '''
