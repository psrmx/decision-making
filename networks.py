import circuits as cir
from brian2 import seed, Network


def get_hierarchical_net(task_info):
    """
    Construct hierarchical net for decision making experiment.

    :return: Brian Network object to run and monitors for plotting
    """
    # decision circuit
    dec_groups, dec_synapses, dec_subgroups = cir.mk_dec_circuit(task_info)

    # sensory circuit
    sen_groups, sen_synapses, sen_subgroups = cir.mk_sen_circuit(task_info)

    # ff and fb synapses
    fffb_synapses = cir.mk_fffb_synapses(task_info, dec_subgroups, sen_subgroups)

    # initial conditions
    seed()
    dec_groups, sen_groups = cir.set_init_conds(task_info, dec_groups, sen_groups)

    # create monitors
    monitors = cir.mk_monitors(task_info, dec_subgroups, sen_subgroups, dec_groups, sen_groups)

    # define network
    net = Network(dec_groups.values(), dec_synapses.values(),
                  sen_groups.values(), sen_synapses.values(),
                  fffb_synapses.values(), *monitors, name='hierarchicalnet')

    return net, monitors


def get_plasticity_net(task_info):
    """
    Construct sensory circuit for inhibitory plasticity experiment.

    :return: Brian Network object to run and monitors for plotting
    """



    # define network
    net = Network(sen_groups.values(), sen_synapses.values(),
                  fffb_synapses.values(), *monitors, name='hierarchicalnet')

    return net, monitors