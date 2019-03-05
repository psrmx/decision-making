from snep.configuration import config
config['type'] = 'brian'
from snep.experiment import Experiment
from snep.library.default.default_brian import run as run_task_brian
import os

timeout = None

class JobInfoExperiment(Experiment):
    run_task = staticmethod(run_task_brian)
    def _prepare_tasks(self):
        from snep.utils import Parameter
        tables = self.tables
        
        rt = config['rt']
        runtime = Parameter(rt, 'second')  # 800
        dt = Parameter(0.1, 'ms')
        target = 'standalone' # 'numpy' # 'weave' # 
        tables.set_simulation(runtime, dt, target)
    
        NE = 8000          # Number of excitatory cells
        NI = NE/4          # Number of inhibitory cells
        lif_params = dict(  tau_ampa = Parameter(5., 'ms'),    # Glutamatergic synaptic time constant
                            tau_gaba = Parameter(10., 'ms'),   # GABAergic synaptic time constant
                            gl = Parameter(10.0,'nsiemens'),   # Leak conductance
                            el = Parameter(-60,'mV'),          # Resting potential
                            er = Parameter(-80,'mV'),          # Inhibitory reversal potential
                            vt = Parameter(-50.,'mV'),         # Spiking threshold
                            memc = Parameter(200.0,'pfarad'),  # Membrane capacitance
                            bgcurrent = Parameter(200,'pA'),)   # External current
        
        eqs_neurons='''
        dv/dt=(-gl*(v-el)-(g_ampa*v+g_gaba*(v-er))+bgcurrent)/memc : volt (unless refractory)
        dg_ampa/dt = -g_ampa/tau_ampa : siemens
        dg_gaba/dt = -g_gaba/tau_gaba : siemens
        '''
        tables.add_neurongroup('neurons', NE+NI, eqs_neurons, lif_params, 'v > vt', 'v = el', refractory='5*ms')
        tables.add_subgroup('exc', 'neurons', 0, NE)
        tables.add_subgroup('inh', 'neurons', NE, NI)
    
        syn_namespace = dict(epsilon = 0.02)
        tables.add_synapse('e2n', 'exc', 'neurons', syn_namespace, pre='g_ampa += 0.3*nS', connect='rand()<epsilon')
        tables.add_synapse('i2i', 'inh', 'inh', syn_namespace, pre='g_gaba += 3.0*nS', connect='rand()<epsilon')
    
        eqs_stdp_inhib = '''
        w : 1
        dA_pre/dt=-A_pre/tau_stdp : 1 (event-driven)
        dA_post/dt=-A_post/tau_stdp : 1 (event-driven)
        '''
        rho0 = 3 # Hz
        tau_stdp = 20 # ms
        i2e_namespace = dict(eta = 1e-2,
                             epsilon = 0.02,                # Sparseness of synaptic connections
                             alpha = rho0*tau_stdp*1e-3*2,
                             gmax = 100,                    # Maximum inhibitory weight
                             tau_stdp = Parameter(tau_stdp,'ms'))    # STDP time constant
        tables.add_synapse('i2e', 'inh', 'exc', 
                           i2e_namespace, 
                           model=eqs_stdp_inhib, 
                           pre='''A_pre += 1.
                                 w = clip(w+(A_post-alpha)*eta, 0, gmax)
                                 g_gaba += w*nS''',
                           post='''A_post += 1.
                                  w = clip(w+A_pre*eta, 0, gmax)
                               ''',
                           connect='rand()<epsilon')
        tables.add_synapse_state_variable_setters('i2e', {'w': Parameter(1e-10),})
    
        self.add_monitors_spike_times({'exc', 'inh'})
        
        self.add_ranges(tables)
    
    def add_ranges(self, tables):
        from snep.utils import ParameterArray
        param_ranges =  {'ng':{'neurons':{'gl':ParameterArray([9,], 'nsiemens')}},#10,11], 'nsiemens')}},#
                         'sy':{'i2e':{'eta':ParameterArray([3e-2,2e-2, ])}}, #1e-2])}}, #
                         'x':ParameterArray([0,1,])}#2])}#
        tables.add_parameter_ranges(param_ranges)
        
        tables.link_parameter_ranges([('sy','i2e','eta'), ('x',)])

if __name__ == '__main__':
    from snep.parallel import run
    '''
    IMPORTANT: Only include code here that can be run repeatedly,
    because this will be run once in the parent process, and then
    once for every worker process.
    '''
    ji_kwargs = dict(root_dir=os.path.expanduser('~/experiments'))
    job_info = run(JobInfoExperiment, ji_kwargs, timeout)
