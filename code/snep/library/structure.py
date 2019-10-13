from ..utils import ParameterArray

def compute_connectivity_delays_old(conns, n_pop, distance_measure, cell_positions, num_target_cells_ijs,
                          connectivity_profile_ijs, propagation_rate_ijs, monosynaptic):
    '''
    Computes connectivity and delay matrices between two populations.

    n_pop - a dictionary of population sizes.
    distance_measure - a function which takes an array of target cell positions, and a position
                        for the source cell and returns an array of distances.
    cell_positions - a dictionary mapping from population names to arrays of cell positions.
    num_target_cells_ijs - a dictionary from (source,target) tuples to the mean number of target
                            cells for every cell in the source population.
    connectivity_profile_ijs - a dictionary mapping from source population name to functions
                             that take an array of distances and return an array of connection 
                             probabilities.
    propagation_rate_ijs - a dictionary from source population name to the axonal propagation velocity
                in metres per second.
    monosynaptic - a dictionary from (source,target) tuples to a boolean indicating if  
                    connections between source and target cells should always be monosynaptic.
    '''
    import numpy as np
    from random import sample
    import matplotlib.pyplot as plt

    connections = {}
    delays = {}
    for (source, target) in conns:
        k_ij = num_target_cells_ijs[(source, target)]
        mono = monosynaptic[(source, target)]
        n_i, n_j = n_pop[source], n_pop[target]

        num_connections_needed_all = np.zeros(n_i, dtype=np.int32) # predefine for cast to int
        num_connections_needed_all[:] = np.random.normal(k_ij, k_ij/20., size=n_i)
        
        connectivity = np.zeros((n_i, n_j), dtype=np.int32)
        propagationtime = np.zeros((n_i, n_j))
        proprate_m_per_ms = propagation_rate_ijs[(source, target)]
        ii = np.arange(n_i)
        for i in ii:
            num_connections_needed = num_connections_needed_all[i]
            num_already_connected = np.sum(connectivity, 1)[i]
            while num_already_connected < num_connections_needed:
                missing_conns = num_connections_needed - num_already_connected
                if mono:  # Only sample from the unconnected targets
                    unconnected = np.where(connectivity[i, :] == 0)[0]
                    target_idx = np.random.permutation(unconnected)
                else:
                    target_idx = np.random.randint(0, n_j, missing_conns*100)
                target_pos = cell_positions[target][target_idx]
                source_pos = cell_positions[source][i]
                distances = distance_measure(target_pos, source_pos)

                P_x = connectivity_profile_ijs[(source, target)](distances)
                new_connections = np.argwhere(P_x >= 1. - np.random.uniform(size=P_x.size)).flatten()
                connected_idx = target_idx[new_connections]
                connected_idx = connected_idx[:missing_conns]

                if mono:
                    connectivity[i, connected_idx] = 1
                else:
                    bins_idx = np.arange(np.max(connected_idx)+1)
                    num_connections_per_neuron = np.bincount(connected_idx)
                    connectivity[i, bins_idx] += num_connections_per_neuron

                # Autosynapses must be zeroed here because otherwise we'll terminate early
                if source == target:
                    connectivity[ii, ii] = 0
                num_already_connected = np.sum(connectivity, 1)[i]

                connected_idx = np.argwhere(connectivity[i, :]).flatten()
                propdist_m = np.abs(distance_measure(cell_positions[target][connected_idx], source_pos))
                proptime_ms = propdist_m / proprate_m_per_ms
                propagationtime[i, connected_idx] = proptime_ms

        # plt.imshow(connectivity)
        # plt.show()
        connections[(source, target)] = ParameterArray(connectivity)
        delays[(source, target)] = ParameterArray(propagationtime, 'ms')
    return delays, connections


def compute_connectivity_delays(conns, n_pop, distance_measure, cell_positions, num_target_cells_ijs,
                connectivity_profile_ijs, propagation_rate_ijs, monosynaptic, scales):
    from scipy import stats
    from scipy.sparse import lil_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    connections = {}
    delays = {}
    for (source, target) in conns:
        print("Starting {0}".format((source, target)))
        k_ij, s_ij = num_target_cells_ijs[(source, target)]
        mono = monosynaptic[(source, target)]
        n_i, n_j = n_pop[source], n_pop[target]
        scale = scales[(source, target)]
        # distances = distance_measure[(source, target)]
        num_connections_needed_all = np.zeros(n_i, dtype=np.uint16)  # predefine for cast to int
        num_connections_needed_all[:] = np.random.normal(k_ij, s_ij, size=n_i) if s_ij != 0 else k_ij

        connectivity = lil_matrix((n_i, n_j), dtype=np.float64)  # np.zeros((n_i, n_j))  #, dtype=np.uint8)
        # all_probs = lil_matrix((n_i, n_j), dtype=np.float32)  # np.zeros((n_i, n_j))  #, dtype=np.float32)
        target_indices = np.arange(n_j)  # , dtype=np.uint16)
        propagation_time = lil_matrix((n_i, n_j), dtype=np.float32)  # np.zeros((n_i, n_j))  #, dtype=np.float32)
        proprate_m_per_ms = propagation_rate_ijs[(source, target)]
        connectivity_i = np.empty(n_j)
        ii = np.arange(n_i)
        for i in ii:
            # print "{0} {1}".format((source, target), i)
            connectivity_i.fill(0)
            num_connections_needed = num_connections_needed_all[i]
            num_already_connected = 0  # connectivity[i, :].sum()
            p_conn = connectivity_profile_ijs[(source, target)]

            source_pos = cell_positions[source][i]
            distances = distance_measure(cell_positions[target], source_pos)
            pk = p_conn(distances)
            support_idx = np.argwhere(pk > 1e-6)
            if source == target:  # Autosynapses have zero probability
                pk[i] = 0
            # all_probs[i, :] = pk
            pk = pk[support_idx].flatten()
            pk /= pk.sum()
            bounded_support = target_indices[support_idx].flatten()
            p = stats.rv_discrete(values=(bounded_support, pk), name="Foo")

            while num_already_connected < num_connections_needed:
                missing_conns = num_connections_needed - num_already_connected
                connected_idx = p.rvs(size=20*missing_conns)
                # connected_idx = bounded_support[connected_idx]
                if mono:
                    connected_idx = np.setdiff1d(connected_idx, np.argwhere(connectivity_i).flatten())
                    # assert (connected_idx == np.unique(connected_idx)).all()
                    np.random.shuffle(connected_idx)
                    connected_idx = connected_idx[:missing_conns]
                    connectivity_i[connected_idx] = 1
                else:
                    connected_idx = connected_idx[:missing_conns]
                    bins_idx = np.arange(np.max(connected_idx)+1)
                    num_connections_per_neuron = np.bincount(connected_idx)
                    connectivity_i[bins_idx] += num_connections_per_neuron

                num_already_connected = connectivity_i.sum()  # np.sum(connectivity, 1)[i]
                if num_already_connected < num_connections_needed:
                    print("Redraw")

            connected_idx = np.argwhere(connectivity_i).flatten()
            connectivity[i, connected_idx] = scale * connectivity_i[connected_idx]
            propdist_m = np.abs(distances[connected_idx])
            proptime_ms = propdist_m / proprate_m_per_ms
            propagation_time[i, connected_idx] = proptime_ms

        print("Finished {0}, copying matrix".format((source, target)))
        connectivity = csr_matrix(connectivity)
#         plt.figure()
#         plt.imshow(connectivity.todense(), interpolation='nearest')
#         #all_probs = csr_matrix(all_probs)
#         #plt.imshow(all_probs.todense(), interpolation='nearest')
#         #plt.figure()
#         plt.show()
        connections[(source, target)] = ParameterArray(connectivity)
        delays[(source, target)] = ParameterArray(propagation_time, 'ms')
    return delays, connections


def compute_connectivity_delays_fixed_in(conns, n_pop, distance_measure, cell_positions, num_inputs_ijs,
                connectivity_profile_ijs, propagation_rate_ijs, monosynaptic, w_scales):
    from scipy import stats
    from scipy.sparse import lil_matrix
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix
    connections = {}
    delays = {}
    zero_threshold = 1e-6
    for (source, target) in conns:
        print("Starting {0}".format((source, target)))
        k_ij, s_ij = num_inputs_ijs[(source, target)]
        mono = monosynaptic[(source, target)]
        n_i, n_j = n_pop[source], n_pop[target]
        w_scale = w_scales[(source, target)]

        connectivity = lil_matrix((n_i, n_j), dtype=np.float64)  # np.zeros((n_i, n_j))  #, dtype=np.uint8)
        all_probs = np.empty((n_i, n_j))  # lil_matrix((n_i, n_j), dtype=np.float64)  # , dtype=np.float32)
        # propagation_time = lil_matrix((n_i, n_j), dtype=np.float32)  # np.zeros((n_i, n_j))  #, dtype=np.float32)
        # proprate_m_per_ms = propagation_rate_ijs[(source, target)]
        for i in xrange(n_i):
            distances = distance_measure(cell_positions[target], cell_positions[source][i])
            all_probs[i, :] = connectivity_profile_ijs[(source, target)](distances)

        num_connections_needed_all = np.zeros(n_j, dtype=np.uint16)  # predefine for cast to int
        num_connections_needed_all[:] = np.random.normal(k_ij, s_ij, size=n_j) if s_ij > 0 else k_ij
        source_indices = np.arange(n_i)
        connectivity_j = np.empty(n_i)
        for j in xrange(n_j):
            # print "{0} {1}".format((source, target), i)
            connectivity_j.fill(0)
            num_connections_needed = num_connections_needed_all[j]
            num_already_connected = 0
            pk = all_probs[:, j]
            support_idx = np.argwhere(pk > zero_threshold).flatten()
            pk[np.argwhere(pk <= zero_threshold)] = 0
            if source == target:  # Autosynapses have zero probability
                pk[j] = 0
            bounded_support = source_indices[support_idx].flatten()
            assert (bounded_support == support_idx).all()
            pk = pk[support_idx].flatten()
            pk /= pk.sum()
            p = stats.rv_discrete(values=(bounded_support, pk), name="Arbitrary-discrete")

            while num_already_connected < num_connections_needed:
                missing_conns = num_connections_needed - num_already_connected
                connected_idx = p.rvs(size=20*missing_conns)
                if mono:
                    connected_idx = np.setdiff1d(connected_idx, np.argwhere(connectivity_j).flatten())
                    np.random.shuffle(connected_idx)
                    connected_idx = connected_idx[:missing_conns]
                    connectivity_j[connected_idx] = 1
                else:
                    connected_idx = connected_idx[:missing_conns]
                    bins_idx = np.arange(np.max(connected_idx)+1)
                    num_connections_per_neuron = np.bincount(connected_idx)
                    connectivity_j[bins_idx] += num_connections_per_neuron

                num_already_connected = connectivity_j.sum()  # np.sum(connectivity, 1)[i]
                if num_already_connected < num_connections_needed:
                    print("Redraw {0}".format(j))

            connected_idx = np.argwhere(connectivity_j).flatten()
            if isinstance(w_scale, tuple):
                ws = np.random.lognormal(mean=w_scale[0], sigma=w_scale[1], size=connected_idx.size)
                connectivity[connected_idx, j] = ws.reshape(connected_idx.size, 1)
            else:
                col_j =  connectivity_j[connected_idx].reshape(connected_idx.size, 1)
                connectivity[connected_idx, j] = w_scale * col_j
            # propdist_m = np.abs(distances[connected_idx])
            # proptime_ms = propdist_m / proprate_m_per_ms
            # propagation_time[connected_idx, j] = proptime_ms

        print("Finished {0}, copying matrix".format((source, target)))
        connectivity = csr_matrix(connectivity)
        # all_probs = csr_matrix(all_probs)
        if 0:
            from snep.utils import plot_connections
            plot_connections(connectivity, all_probs)
        connections[(source, target)] = ParameterArray(connectivity)
        # delays[(source, target)] = ParameterArray(propagation_time, 'ms')
    return connections


def compute_connectivity_brian2(N_pre, N_post, pos_pre, pos_post, distance_ij, connectivity_at_distance, 
                                input_mu, input_std, autosynapses):
    from scipy import stats
    from scipy.sparse import lil_matrix
    import numpy as np
    from snep.configuration import on_cluster
    zero_threshold = 1e-6
    do_plot = False if on_cluster else 0

    print("Starting connectivity {0}x{1}".format(N_pre, N_post))

    all_probs = np.empty((N_pre, N_post))  # lil_matrix((N_pre, N_post), dtype=np.float64)  # , dtype=np.float32)
    idx_j = np.arange(N_post)
    for i in xrange(N_pre):
        distances = distance_ij(pos_post, pos_pre, idx_j, i)
        all_probs[i, :] = connectivity_at_distance(distances)

    num_connections_needed_all = np.zeros(N_post, dtype=np.uint16)  # predefine for cast to int
    num_connections_needed_all[:] = np.random.normal(input_mu, input_std, size=N_post) if input_std > 0 else input_mu
    source_indices = np.arange(N_pre)
    conn_pre, conn_post = [], []
    for j in xrange(N_post):
        pk = all_probs[:, j]
        support_idx = np.argwhere(pk > zero_threshold).flatten()
        pk[np.argwhere(pk <= zero_threshold)] = 0
        if not autosynapses:  # Autosynapses have zero probability
            pk[j] = 0
        bounded_support = source_indices[support_idx].flatten()
        assert (bounded_support == support_idx).all()
        pk = pk[support_idx].flatten()
        pk /= pk.sum()
        p = stats.rv_discrete(values=(bounded_support, pk), name="Arbitrary-discrete")

        num_connections_needed = num_connections_needed_all[j]
        num_already_connected = 0
        connected_pre = []
        while num_already_connected < num_connections_needed:
            missing_conns = num_connections_needed - num_already_connected
            candidate_idx = p.rvs(size=20*missing_conns)
            # remove repeated indices
            candidate_idx = np.unique(candidate_idx)
            # remove indices already connected 
            candidate_idx = np.setdiff1d(candidate_idx, connected_pre)
            # shuffle result because it's sorted
            np.random.shuffle(candidate_idx)
            # take the necessary number of indices
            candidate_idx = candidate_idx[:missing_conns]
            connected_pre += candidate_idx.tolist()

            num_already_connected = len(connected_pre)
            if num_already_connected < num_connections_needed:
                print("Redraw {0}".format(j))

        conn_post += [j] * num_connections_needed
        conn_pre += connected_pre

    if do_plot:
        from snep.utils import plot_connections
        import matplotlib.pyplot as plt
        connectivity = lil_matrix((N_pre, N_post))
        connectivity[conn_pre, conn_post] = 1
        in_degree = connectivity.sum(axis=0)
        out_degree = connectivity.sum(axis=1)
        print('IN mu: {} std: {}'.format(in_degree.mean(), in_degree.std()))
        print('OUT mu: {} std: {}'.format(out_degree.mean(), out_degree.std()))
        plot_connections(connectivity, all_probs)
        plt.show()
    print("Finished connecting")

    return np.array(conn_pre), np.array(conn_post)


def make_connections_fixed_in(N_pre, N_post, conn_params):
    import numpy as np
    pre_dist = conn_params['pre_dist']
    post_dist  = conn_params['post_dist']
    circular = conn_params['circular']

    dist_ratio = np.float64(pre_dist) / np.float64(post_dist)
    pos_pre = np.arange(N_pre) * pre_dist
    pos_post = np.arange(N_post) * post_dist + dist_ratio

    dist_pre, dist_post = np.empty(N_pre), np.empty(N_post)
    if circular:
        def distance_ij(tar_pos, src_pos, tar_j, src_i):
            tar_pos, src_pos = tar_pos[tar_j], src_pos[src_i]
            dist = dist_pre if tar_pos.size == N_pre else dist_post
            space = tar_pos.max() - tar_pos.min() + np.abs(tar_pos[0] - tar_pos[1])
            np.subtract(tar_pos, src_pos, out=dist)
            lower, upper = -space/2, space/2
            too_neg, too_pos = dist < lower, dist > upper
            dist[too_neg] = dist[too_neg] - lower + upper
            dist[too_pos] = dist[too_pos] - upper + lower
            return dist
    else:
        def distance_ij(tar_pos, src_pos, tar_j, src_i):
            assert False, 'No implementation for distance_ij in linear network'

    if conn_params['profile'] == 'exponential':
        scale = conn_params['scale']
        P_e, P_i = np.empty(N_pre), np.empty(N_post)
        con = lambda d: np.clip(np.exp(-np.abs(d)/scale), 0., 1., out=(P_e if d.size == P_e.size else P_i))
    elif conn_params['profile'] == 'gaussian':
        assert False, 'Gaussian connection profile not implemented'
    else:
        assert False, 'Unknown connection profile'

    input_mu = N_pre * conn_params['p']
    if conn_params['fixed_in']:
        input_std = 0.
    else:
        input_std = np.sqrt(N_pre * conn_params['p'] * (1. - conn_params['p']))

    autosynapse = conn_params['auto']
    monosynapse = conn_params['mono']
    assert monosynapse
    return compute_connectivity_brian2(N_pre, N_post, pos_pre, pos_post, distance_ij, con, input_mu, input_std, autosynapse)


def linear_absolute_positions(name_py, name_in, n_pop, intercellular_dist):
    '''
    This will construct a linear array of cell positions with N pyramidal cells for
    every 1 interneuron. All cells are intercellular_dist apart from each other.
    The indices for the two cell types are returned as arrays in the dictionary idx.
    '''
    import numpy as np
    n_py = n_pop[name_py]
    n_in = n_pop[name_in]
    assert(not n_py % n_in)
    idx = {}
    n_pyr_per_subunit = n_py / n_in

    pos = np.arange(1,n_in+n_py+1) * intercellular_dist
    idx[name_in] = np.arange(n_pyr_per_subunit,n_py+n_in,n_pyr_per_subunit+1)
    py_trick = np.ones(n_py+n_in)
    py_trick[idx[name_in]] = 0
    idx[name_py] = np.where(py_trick)
        
    return pos, idx

