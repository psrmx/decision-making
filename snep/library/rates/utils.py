import matplotlib.pyplot as plt
import numpy as np
from ...utils import reallocate_aligned
# from numba import jit
from .inner import dtype
from collections import namedtuple

cache = False
PLOT_CONNECTIONS = 0
VON_MISES_CONNECTIONS = 1
csr_indices = namedtuple('csr_indices', ['indptr', 'indices', 'data'])
connection_sparse = namedtuple('connection_sparse', ['csr_indices', 'row_ind', 'col_ind', 'conn_prob'])
connection_dense = namedtuple('connection_dense', ['array', 'row_ind', 'col_ind', 'conn_prob'])

def compute_connections(sparse, scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                        scalar_data, j_scale=None, j_var=0., in_degree_cv=-1.):
    if sparse:
        return _compute_connections_sparse(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                                           scalar_data, j_scale, j_var, in_degree_cv)
    else:
        return _compute_connections_dense(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                                          j_scale, j_var, in_degree_cv)


def convert_int16(x):
    astype = np.int16
    int_max = np.iinfo(astype).max
    c, row_ind, col_ind = x
    if isinstance(c, tuple):
        indptr, indices, data = c
        if indptr.max() < int_max:
            indptr = indptr.astype(astype)
        if indices.max() < int_max:
            indices = indices.astype(astype)
        c = indptr, indices, data

    if row_ind.max() < int_max:
        row_ind = row_ind.astype(astype)
    if col_ind.max() < int_max:
        col_ind = col_ind.astype(astype)
    return c, row_ind, col_ind


# @jit(nopython=True, cache=cache)
def _compute_connections_sparse(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                                scalar_data, J_scale, J_var, in_degree_cv):
    """
    where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
    """
    conns, coo, cp = compute_connections_base(scale, p, f_global, d_pre, N_pre, d_post, N_post, J,
                                              zero_diagonal, J_scale, J_var, in_degree_cv)

    row_ind = reallocate_aligned(coo.row)
    col_ind = reallocate_aligned(coo.col)

    csr = coo.tocsr()
    indptr = reallocate_aligned(csr.indptr)
    indices = reallocate_aligned(csr.indices)
    data = reallocate_aligned(csr.data)

    if scalar_data:
        data = J
        assert False
    # return (indptr, indices, data), row_ind, col_ind
    return connection_sparse(csr_indices(indptr, indices, data), row_ind, col_ind, cp)


def _compute_connections_dense(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                               J_scale, J_var, in_degree_cv):
    conns, coo, cp = compute_connections_base(scale, p, f_global, d_pre, N_pre, d_post, N_post, J,
                                              zero_diagonal, J_scale, J_var, in_degree_cv)
    conns = reallocate_aligned(conns)
    row_ind = reallocate_aligned(coo.row)
    col_ind = reallocate_aligned(coo.col)
    # return conns, row_ind, col_ind
    return connection_dense(conns, row_ind, col_ind, cp)


def compute_connections_base(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal,
                             J_scale, J_var, in_degree_cv):
    from scipy.sparse import coo_matrix
    if VON_MISES_CONNECTIONS:
        pi_scale = 1. / (np.pi * scale)**2
    else:
        pi_scale = scale * np.pi
    d_min, d_max = -np.pi, np.pi
    pos_pre = np.linspace(d_min, d_max, N_pre, endpoint=False)
    pos_post = np.linspace(d_min, d_max, N_post, endpoint=False)

    fixed_in = in_degree_cv == 0.

    k = p*(N_pre - 1) if zero_diagonal else p*N_pre
    if in_degree_cv is None or in_degree_cv < 0.:
        sd_k = np.sqrt(k)
        in_degree_cv = sd_k / k
        print('CV of in-degree not specified, using CV {:1.2f} for K {:.1f}'.format(in_degree_cv, k))

    if not fixed_in:
        sd_k = in_degree_cv * k
        c = np.random.normal(loc=k, scale=sd_k, size=N_post)
        c = np.round(c).astype(np.int)
        c_zero = c <= 0
        print('Number of zero in-degrees', c_zero.sum())
        c[c_zero] = 1
    else:
        c0 = int(np.round(k))
        print('Fixed in-degree of {}'.format(c0))
        c = np.ones(N_post, dtype=np.int) * max(1, c0)
    use_choice = 1  # Don't change this unless you're REALLY sure.

    conns = np.zeros((N_post, N_pre), dtype=dtype)
    cp = np.zeros((N_post, N_pre), dtype=dtype)
    retried = 0
    for i in range(N_post):
        d_curr = pos_post[i] - pos_pre
        d_curr[d_curr < d_min] -= 2*d_min
        d_curr[d_curr > d_max] -= 2*d_max
        if scale == np.infty:
            ps = np.ones_like(d_curr)
        elif VON_MISES_CONNECTIONS:
            ps = np.exp(pi_scale * (np.cos(d_curr) - 1))
            # ps = vonmises.pdf(d_curr, pi_scale)
        else:
            ps = np.exp(-np.abs(d_curr)/pi_scale)
        if zero_diagonal:
            ps[np.abs(d_curr) < 1e-12] = 0
        ps /= ps.sum() / (1. - f_global)
        ps += f_global / ps.size
        nnz = ps.nonzero()[0].size
        if nnz < c[i]:
            print('WARNING: Fewer pre-synaptic cells with non-zero connection '
                  'probability than desired in-degree! {} {}'.format(nnz, c[i]))
        c_i = min(nnz, c[i])
        cp[i, :] = ps * c_i
        if use_choice:
            # Seems to under-sample near areas of high connection density?
            j = np.random.choice(N_pre, size=c_i, replace=False, p=ps)
        else:
            # This is worse than random.choice, does not produce the required in-degree c_i
            ps *= c_i
            retry = True
            while retry:
                s = np.random.uniform(size=N_pre)
                j = np.argwhere(ps > s)
                # retry = j.size < c_i
                retry = j.size == 0
                if retry:
                    retried += 1
        # if fixed_in:
        #     j = np.random.choice(N_pre, size=c_i, replace=False, p=ps)
        # else:
        #     ps *= c_i
        #     retry = True
        #     while retry:
        #         s = np.random.uniform(size=N_pre)
        #         j = np.argwhere(ps > s)
        #         retry = j.size == 0
        #         if retry:
        #             retried += 1
        conns[i, j] = 1.

    if PLOT_CONNECTIONS:
        plot_connection(scale, p, f_global, d_pre, N_pre, d_post, N_post, zero_diagonal, conns)

    j_rescale = scale_j(conns, p, N_pre, J, J_scale, retried)
    coo = coo_matrix(conns)
    coo.data *= j_rescale
    in_w = coo.sum(axis=1)
    in_mu = in_w.mean()
    in_std = in_w.std()
    print('Input weight CV {}, mean {}, std {}'.format(in_std/in_mu, in_mu, in_std))
    if J_var > 0.:
        print('Before randomization')
        report_connection(conns, p, N_pre, retried)
        j = np.random.normal(loc=j_rescale, scale=np.abs(j_rescale*J_var), size=coo.data.size)
        lt0 = j < 0.
        print("Produced {} < 0".format(lt0.sum()/j.size))
        j[lt0] = 0.
        coo.data[:] = j
        print('After randomization')
        report_connection(coo.toarray(), p, N_pre, retried)
    conns = coo.toarray()

    return conns, coo, cp


def compute_probabilities(p, fraction_global, scale, d_pre, N_pre, zero_diagonal):
    d = np.linspace(-np.pi, np.pi, N_pre, endpoint=False)
    if scale == np.infty:
        ps = np.ones_like(d)
    elif VON_MISES_CONNECTIONS:
        pi_scale = 1. / (np.pi * scale) ** 2
        ps = np.exp(pi_scale * (np.cos(d) - 1))
    else:
        ps = np.exp(-np.abs(d)/(scale * np.pi))
    if zero_diagonal:
        ps[0] = 0
    ps = np.roll(ps, N_pre//2)
    ps /= ps.sum() / (1. - fraction_global)
    ps += fraction_global / ps.size
    return ps


def plot_connection(scale, p, f_global, d_pre, N_pre, d_post, N_post, zero_diagonal, conns):
    from matplotlib import cm
    from matplotlib.pyplot import GridSpec
    from scipy.fftpack import fft, fftfreq, fftshift
    if N_pre >= N_post:
        fig = plt.figure(figsize=(6, 6))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
    else:
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(1, 2, width_ratios=[1, 2])
    axes = [fig.add_subplot(g) for g in gs]
    conns = conns.astype(np.float64)
    axes[0].imshow(conns, interpolation='none', aspect=1., cmap=cm.magma) # vmin=0., vmax=1.,

    roll = 1. if d_pre == d_post else d_post / d_pre
    ps = compute_probabilities(p, f_global, scale, d_pre, N_pre, zero_diagonal)
    c = int(np.round(p * N_pre))
    s = np.zeros(conns.shape[1])
    for i in range(conns.shape[0]):
        s += np.roll(conns[i, :], -int(i*roll)).squeeze()
    # if 0:
    #     identical = np.array([(conns[:, i-1] * conns[:, i]).sum() / conns[:, i-1].sum() for i in range(1, N_pre)])
    # else:
    #     n_nearby = N_pre if scale==np.infty else int(np.round(N_pre * 2*scale))
    #     ref_conns = conns[:, 0].sum()
    #     identical = np.array([(conns[:, i] * conns[:, 0]).sum() / ref_conns for i in range(1, n_nearby)])
    #
    # axes[0].set_title('Average overlap f: {:.3f}, sd: {:.2f}'.format(identical.mean(), identical.std()))


    axes[1].plot(s)  # / conns.shape[0]
    axes[1].plot(ps*c*N_post)  # / ps.max())  #
    axes[1].set_xlim(-5, conns.shape[1]+5)
    axes[1].hlines(conns.shape[0], *axes[1].get_xlim(), color='g')

    if 0:
        ps_fd = np.abs(fft(ps))
        ps_fr = fftfreq(ps.size, 1 / ps.size)
        axes[2].plot(fftshift(ps_fr), fftshift(ps_fd))


def compute_connections_overlap(p, N_pre, N_post, J):
    from scipy.sparse import coo_matrix

    if 1:
        conns = np.zeros((N_post, N_pre), dtype=dtype)
        overlapped = int(np.round(p * N_post))
        single = int(np.round((1-p)*N_post)) // 2
        conns[:single+overlapped, 0]  = 1
        conns[-(single+overlapped):, 1] = 1
    else:
        no_overlap = N_post // N_pre
        full_overlap = N_post
        delta_overlap = full_overlap - no_overlap
        per_cell = int(np.round(p*delta_overlap)) + no_overlap
        conn = np.zeros(N_post)
        conn[:per_cell] = 1
        conns = np.zeros((N_post, N_pre), dtype=dtype)
        for i in range(N_pre):
            conns[:, i] = np.roll(conn, i*no_overlap)

    plot_connection(np.infty, p, 0, 4, 2, 1, N_post, 0, conns)

    ind = conns.sum(axis=1)
    in_mu = ind.mean()
    if 0:
        conns *= J / in_mu
    else:
        conns *= J / N_pre
    ind = conns.sum(axis=1)
    print(J, in_mu, ind.mean(), ind.std())

    coo = coo_matrix(conns)

    row_ind = reallocate_aligned(coo.row)
    col_ind = reallocate_aligned(coo.col)

    csr = coo.tocsr()
    indptr = reallocate_aligned(csr.indptr)
    indices = reallocate_aligned(csr.indices)
    data = reallocate_aligned(csr.data)

    return connection_sparse(csr_indices(indptr, indices, data), row_ind, col_ind)


def compute_kernel(sp_scale, d_exc, n_e, shape, fft=False):
    from numpy.fft.fftpack import rfft, irfft
    if np.infty == sp_scale:
        diffusion_kernel = np.ones(n_e, dtype=dtype)
        diffusion_kernel /= diffusion_kernel.sum()
    elif 1e-3 < sp_scale:
        length = d_exc*n_e
        d = np.linspace(d_exc-length/2, length/2, n_e, dtype=dtype)
        d = np.roll(d, n_e//2+1)
        if sp_scale == np.infty:
            k = np.ones_like(d)
        elif 'exponential' == shape:
            k = np.exp(-np.abs(d)/sp_scale)
        elif 'gaussian' == shape:
            k = np.exp(d**2 / (-2*sp_scale**2))
        elif 'vonmises':
            std_pi = sp_scale / (n_e / 2) * np.pi
            kappa = 1. / std_pi**2
            x = np.pi * d / d.max()
            k = np.exp(kappa * (np.cos(x) - 1))
            k -= k.min()
            # k = vonmises.pdf(x, kappa)
        else:
            assert False, "Unknown diffusion kernel type"
        k /= k.sum()
        assert np.isfinite(k).all(), "Diffusion scale may be too small for this network size"
        diffusion_kernel = rfft(k) if fft else k
        if 0:
            r = np.zeros_like(k)
            r[0] = 1
            x = irfft(rfft(k) * rfft(r)).real
            print(sp_scale, x.sum(), k.sum())
            plt.figure()
            plt.plot(x, '-x')
            plt.plot(k+.0001)
            plt.show()
    else:
        diffusion_kernel = np.zeros(n_e, dtype=dtype)
        diffusion_kernel[0] = 1.
    diffusion_kernel = reallocate_aligned(diffusion_kernel)
    return diffusion_kernel


def compute_heterogeneity(n, h_type, h_scale, h_width, h_loc):
    if h_type == 'local':
        ret = np.zeros(n, dtype=dtype)
        ret[h_loc-h_width//2:h_loc+h_width//2] = h_scale
    elif h_type == 'vonmises':
        if h_width < np.infty:
            pi_width = 1. / (np.pi * h_width / (n/2))**2
            pi_loc = 2*np.pi * h_loc / n  - np.pi
            d = np.linspace(-np.pi, np.pi, n, endpoint=False, dtype=dtype)
            ret = np.exp(pi_width * (np.cos(d - pi_loc) - 1))
            ret -= ret.min()
        else:
            ret = np.ones(n)
        if h_scale > 0.:
            ret /= ret.max() / h_scale
        else:
            ret[:] = 0.
    elif h_type == 'global':
        ret = np.ones(n, dtype=dtype) * h_scale
    else:
        assert False, "Unknown heterogeneity type"
    # plt.plot(ret)
    # plt.show()
    return ret


def remake_connections(sparse, row_ind, col_ind, N_pre, N_post, J, scalar=False):
    if sparse:
        return _remake_connections_sparse(row_ind, col_ind, N_pre, N_post, J, scalar)
    else:
        return _remake_connections_dense(row_ind, col_ind, N_pre, N_post, J)


def _remake_connections_sparse(row_ind, col_ind, N_pre, N_post, J, scalar):
    from scipy.sparse import coo_matrix
    assert not scalar, "Not implemented for scalar data"

    if np.isscalar(J):
        data = np.empty(row_ind.size, dtype=dtype)
        data.fill(J)
    else:
        data = J
    conns = coo_matrix((data, (row_ind, col_ind)), shape=(N_post, N_pre))
    csr = conns.tocsr()

    if PLOT_CONNECTIONS:
        plt.figure()
        plt.imshow(conns.toarray())

    data = csr.data
    if scalar and np.isscalar(J):
        data = J
    else:
        data = reallocate_aligned(data)

    return csr_indices(reallocate_aligned(csr.indptr), reallocate_aligned(csr.indices), data)


def _remake_connections_dense(row_ind, col_ind, N_pre, N_post, J):
    from scipy.sparse import coo_matrix
    if np.isscalar(J):
        data = np.empty(row_ind.size, dtype=dtype)
        data.fill(J)
    else:
        data = np.squeeze(J)
    conn = coo_matrix((data, (row_ind, col_ind)), shape=(N_post, N_pre))
    return reallocate_aligned(conn.toarray())


def report_connection(conns, p, N_pre, retried):
    a0s, a1s = conns.sum(axis=0), conns.sum(axis=1)
    calc_mu = p*N_pre
    calc_var = p * (1 - p) * N_pre
    calc_std = np.sqrt(calc_var)  # np.sqrt(p_N_pre)
    std_scale = a1s.std()
    mu_scale = a1s.mean()
    var_scale = a1s.var()
    conn_str = 'N_post: {: 4d} N_pre: {: 4d} retried: {: 3d} '.format(conns.shape[0], conns.shape[1], retried)
    out_str = 'out-degree mu: {: 6.1f} std: {: 4.1f} min: {: 4.0f} '.format(a0s.mean(), a0s.std(), a0s.min())
    in_str = 'in-degree cv: {:0.2f} mu: {: 6.1f} ({: 6.1f}) std: {: 4.2f} ({: 6.2f}) var: {: 4.2f} min: {: 4.0f}'.format(
        std_scale/mu_scale, mu_scale, calc_mu, std_scale, calc_std, var_scale, a1s.min())
    for s in (80*'=', conn_str, out_str, in_str, 80*'='):
        print(s)
    return calc_mu, calc_std, calc_var, mu_scale, std_scale, var_scale


def scale_j(conns, p, N_pre, J, J_scale, retried):
    calc_mu, calc_std, calc_var, mu_scale, std_scale, var_scale = report_connection(conns, p, N_pre, retried)
    if J_scale == 'mu':
        J_calc, J_emp = J / calc_mu, J / mu_scale
        print("J/p*N: {} J/mu: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
        j_rescale = J / mu_scale
    elif J_scale == 'std':
        J_calc, J_emp = J/calc_std, J/std_scale
        print("J/sqrt(p*(1-p)*N): {} J/std: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
        j_rescale = J / std_scale
    elif J_scale == 'var':
        J_calc, J_emp = J/calc_var, J/var_scale
        print("J/p*(1-p)*N: {} J/var: {} diff*N_pre/J: {}".format(J_calc, J_emp, (J_calc-J_emp)*N_pre/J))
        j_rescale = J / var_scale
    else:
        j_rescale = J
    return j_rescale


def soft_plus_thresh_gain(alpha, x, thresh=None, gain=None):
    if thresh is not None:
        y = x - thresh
    else:
        y = x.copy()
    y *= alpha
    y = np.exp(y)
    y = np.log1p(y)
    y /= alpha
    inf_i = np.logical_not(np.isfinite(y))
    if thresh is not None:
        y[inf_i] = x[inf_i] - thresh
    else:
        y[inf_i] = x[inf_i]
    if gain is not None:
        y *= gain
    return y

def logistic_thresh_gain(alpha, x, thresh=None, gain=None):
    if thresh is not None:
        y = alpha*(thresh - x)
    else:
        y = -alpha*x
    np.exp(y, out=y)
    y += 1
    np.power(y, -1, out=y)
    if gain is not None:
        y *= gain
    return y

# @jit(nopython=True, cache=cache)
def csr_matvec_jit_scalar_data(n_row, n_col, indptr, indices, data, r, h):
    for i in range(n_row):
        s = 0.
        for jj in range(indptr[i], indptr[i+1]):
            s += data * r[indices[jj]]
        h[i] = s


def compute_stimulus_set(stimulus_locations, preferred_stimulus_locations, radial_half_width, peak,
                         peak_ratios=0., zero_offset=False):
    '''
    Computes a set of circular stimulus response curves at the specified stimulus locations.

    :param stimulus_locations: Locations in range (-pi, pi) at which to calculate response of cells.
    :param preferred_stimulus_locations: Location at which each cells exhibits its peak firing rate.
    :param radial_half_width: Half-width of each cell's tuning curve.
    :param peak: Peak firing rate at preferred location.
    :param peak_ratios: Ratio of peak firing rate to the rate at pi radians distance around the circle.
    :return: Returns an array of (num stimuli, num cells) firing rates.
    '''
    n_cells = preferred_stimulus_locations.size
    if np.isscalar(stimulus_locations) and isinstance(stimulus_locations, int):
        n_stimulus_locations = stimulus_locations
        x = np.linspace(-np.pi, np.pi, n_stimulus_locations, endpoint=False, dtype=dtype)
    else:
        n_stimulus_locations = stimulus_locations.size
        x = stimulus_locations
    if np.isscalar(peak) or peak.size == 1:
        peak = peak * np.ones(n_cells, dtype=dtype)
    if np.isscalar(peak_ratios) or peak_ratios.size == 1:
        peak_ratios = peak_ratios * np.ones(n_cells, dtype=dtype)
    if np.isscalar(radial_half_width) or radial_half_width.size == 1:
        kappa = np.ones(n_cells, dtype=dtype)
        kappa *= 1./radial_half_width**2 if radial_half_width > 0 else np.infty
    else:
        kappa = 1. / radial_half_width**2

    assert kappa.size == n_cells and peak.size == n_cells and peak_ratios.size == n_cells

    inputs = np.empty((n_stimulus_locations, n_cells), dtype=dtype)
    for n, mu in enumerate(preferred_stimulus_locations):
        k = np.exp(kappa[n] * (np.cos(x - mu) - 1))
        k[np.isnan(k)] = 1
        k *= peak[n]
        if np.abs(peak_ratios[n]) > 1e-16:
            pr = np.exp(kappa[n] * (np.cos(x - mu - np.pi) - 1))
            pr *= peak[n] * peak_ratios[n]
            k += pr
        inputs[:, n] = k

    if zero_offset:
        inputs -= inputs.min()

    return inputs


def compute_stimulus_locations(dt, time_per_stimulus, frames_per_stimulus, n_stimuli,
                               background_per_pass, no_plastic_passes,
                               stimulus_range_passes):
    computed_locations = namedtuple('stimulus_locations',
                                    ['n_steps', 'steps_per_frame', 'stimulus_switch_n',
                                     'stimulus_location_index',
                                     'plasticity_on_frame', 'plasticity_off_frame'])
    locations_per_pass = n_stimuli + background_per_pass
    steps_per_stimulus = int(np.round(time_per_stimulus / dt))
    frames_per_pass = locations_per_pass * frames_per_stimulus
    steps_per_frame = steps_per_stimulus // frames_per_stimulus
    n_steps = stimulus_range_passes * frames_per_pass * steps_per_frame
    plasticity_on_frame = no_plastic_passes * frames_per_pass
    plasticity_off_frame = n_steps // steps_per_frame - no_plastic_passes * frames_per_pass
    stimulus_switch_n = np.arange(steps_per_stimulus, n_steps + steps_per_stimulus, steps_per_stimulus, dtype=np.int32)
    stimulus_location_index = np.zeros_like(stimulus_switch_n)

    one_pass = np.array(list(range(1, n_stimuli + 1)) + background_per_pass * [0])
    for l in range(stimulus_range_passes):
        li = l*locations_per_pass
        one_pass = np.random.permutation(one_pass)
        stimulus_location_index[li:li + locations_per_pass] = one_pass
    for l in range(n_stimuli + 1):
        if l:
            assert (stimulus_location_index == l).sum() == stimulus_range_passes
        else:
            assert (stimulus_location_index == l).sum() == background_per_pass * stimulus_range_passes

    return computed_locations(n_steps, steps_per_frame, stimulus_switch_n, stimulus_location_index,
                              plasticity_on_frame, plasticity_off_frame)

'''
@jit(nopython=True, cache=cache)
def compute_connections_dense(scale, p, f_global, d_pre, N_pre, d_post, N_post, J, zero_diagonal):
    roll = 1. if d_pre == d_post else d_post / d_pre
    ps = compute_probabilities(p, f_global, scale, d_pre, N_pre, zero_diagonal)
    c = number_of_presynaptic_cells(p, N_pre, N_post, 0.)
    total_synapses = c.sum()
    row_ind = np.empty(total_synapses, dtype=np.int32)
    col_ind = np.empty(total_synapses, dtype=np.int32)
    conns = np.zeros((N_post, N_pre), dtype=dtype)
    lb = 0
    for i in range(N_post):
        ub = lb + c[i]
        j = np.random.choice(N_pre, c[i], False, np.roll(ps, int(i*roll)))
        j.sort()
        row_ind[lb:ub] = i
        col_ind[lb:ub] = j
        conns[i, j] = J
        lb = ub
    if PLOT_CONNECTIONS:
        tmp_conns = np.zeros_like(conns)
        tmp_conns[conns > 0] = 1.
        plot_connection(scale, p, f_global, d_pre, N_pre, d_post, N_post, zero_diagonal, tmp_conns)
    return conns, row_ind, col_ind


def number_of_presynaptic_cells(p, N_pre, N_post, in_std):
    input_mu = int(np.round(p * N_pre))
    c = np.empty(N_post, dtype=np.int32)
    if in_std < 1e-6:
        c.fill(input_mu)
    else:
        # input_std = np.sqrt(N_pre * p * (1. - p))
        c[:] = np.maximum(0, np.round(np.random.normal(input_mu, in_std, size=N_post)))
    return c


# @jit(nopython=True, cache=cache)
def coo_tocsr(n_row, Ai, Aj, Ax):
    # compute number of non-zero entries per row of A coo_tocsr

    nnz = Ax.size
    Bp = np.empty(n_row + 1, dtype=np.int32)
    Bj = np.empty(nnz, dtype=np.int32)
    Bx = np.empty(nnz, dtype=Ax.dtype)

    Bp[:] = 0.  # std::fill(Bp, Bp + n_row, 0);

    for n in range(nnz):
        Bp[Ai[n]] += 1

    # cumsum the nnz per row to get Bp[]
    cumsum = 0
    for i in range(n_row):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp

    Bp[n_row] = nnz

    # write Aj,Ax into Bj,Bx
    for n in range(nnz):
        row = Ai[n]
        dest = Bp[row]

        Bj[dest] = Aj[n]
        Bx[dest] = Ax[n]

        Bp[row] += 1

    last = 0
    for i in range(n_row+1):
        temp = Bp[i]
        Bp[i] = last
        last = temp
    # now Bp,Bj,Bx form a CSR representation (with possible duplicates)
    return Bp, Bj, Bx


# @jit(nopython=True, cache=cache)
def compute_coo(p, f_global, scale, d_pre, N_pre, d_post, N_post, J, zero_diagonal, in_std):
    """
    where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
    """
    roll = 1. if d_pre == d_post else d_post / d_pre
    ps = compute_probabilities(p, f_global, scale, d_pre, N_pre, zero_diagonal)
    c = number_of_presynaptic_cells(p, N_pre, N_post, in_std)
    total_synapses = c.sum()
    row_ind = np.empty(total_synapses, dtype=np.int32)
    col_ind = np.empty(total_synapses, dtype=np.int32)
    lb = 0
    for i in range(N_post):
        # lb, ub = i*c, i*c+c
        ub = lb + c[i]
        j = np.random.choice(N_pre, c[i], False, np.roll(ps, int(i*roll)))
        j.sort()
        row_ind[lb:ub] = i
        col_ind[lb:ub] = j  # np.random.choice(N_pre, c, replace=False)
        lb = ub
    data = np.ones(total_synapses, dtype=dtype) * J

    return row_ind, col_ind, data


@jit(nopython=True, cache=cache)
def hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_i, c_e2i_j, r_inh, r_exc):
    for i in range(c_e2i_data.size):
        c_e2i_data[i] += eta * (r_exc[c_e2i_j[i]] - rho0) * r_inh[c_e2i_i[i]]
    np.maximum(c_e2i_data, 0.0, c_e2i_data)
    np.minimum(c_e2i_data, J_max, c_e2i_data)


@jit(nopython=True, cache=cache)
def non_hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_j, r_exc):
    for i in range(c_e2i_data.size):
        c_e2i_data[i] += eta * (r_exc[c_e2i_j[i]] - rho0)
    np.maximum(c_e2i_data, 0.0, c_e2i_data)
    np.minimum(c_e2i_data, J_max, c_e2i_data)


@jit(nopython=True, cache=cache)
def inner_jit(dt, eta, rho0, n_steps, NE, NI, r_exc, r_inh, in_exc, in_inh, f_max, J_max,
          c_e2i_indptr, c_e2i_indices, c_e2i_data,
          c_i2e_indptr, c_i2e_indices, c_i2e_data,
          c_e2e_indptr, c_e2e_indices, c_e2e_data,
          c_i2i_indptr, c_i2i_indices, c_i2i_data,
          c_e2i_j, c_e2i_i, h_e2i, h_i2e, h_e2e, h_i2i,
          r_step, rates, weights, is_hebbian, is_mean):
    r_hat = np.empty_like(r_exc) if is_mean else r_exc
    for n in range(n_steps):
        csr_matvec_jit(NI, c_i2i_indptr, c_i2i_indices, c_i2i_data, r_inh, h_i2i)
        csr_matvec_jit(NI, c_e2i_indptr, c_e2i_indices, c_e2i_data, r_exc, h_e2i)
        csr_matvec_jit(NE, c_i2e_indptr, c_i2e_indices, c_i2e_data, r_inh, h_i2e)
        csr_matvec_jit(NE, c_e2e_indptr, c_e2e_indices, c_e2e_data, r_exc, h_e2e)

        r_exc += dt*((h_e2e - h_i2e + in_exc) - r_exc)
        r_inh += dt*((h_e2i - h_i2i + in_inh) - r_inh)

        np.maximum(r_exc, 0.0, r_exc)
        np.minimum(r_exc, f_max, r_exc)
        np.maximum(r_inh, 0.0, r_inh)
        np.minimum(r_inh, f_max, r_inh)

        if is_mean:
            r_hat[:] = r_exc.mean()
        else:
            r_hat = r_exc

        if is_hebbian:
            hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_i, c_e2i_j, r_inh, r_hat)
        else:
            non_hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_j, r_hat)

        if not n % r_step:
            i = n / r_step
            rates[i, 0] = r_exc.mean()
            rates[i, 1] = np.power(r_exc, 2).mean()
            rates[i, 2] = r_exc.var()
            weights[i, :] = c_e2i_data.mean(), c_e2i_data.var()

    return rates, weights


@jit(nopython=True, cache=cache)
def inner_dense_jit(dt, eta, rho0, n_steps, NE, NI, r_exc, r_inh, in_exc, in_inh, f_max, J_max,
                c_e2i_indptr, c_e2i_indices, c_e2i_data,
                c_i2e, c_e2e, c_i2i, c_e2i_j, c_e2i_i, h_e2i, h_i2e, h_e2e, h_i2i,
                r_step, rates, weights, is_hebbian, is_mean):
    r_hat = np.empty_like(r_exc) if is_mean else r_exc
    for n in range(n_steps):
        np.dot(c_i2i, r_inh, out=h_i2i)
        np.dot(c_i2e, r_inh, out=h_i2e)
        np.dot(c_e2e, r_exc, out=h_e2e)
        csr_matvec_jit(NI, c_e2i_indptr, c_e2i_indices, c_e2i_data, r_exc, h_e2i)

        r_exc += dt*((h_e2e - h_i2e + in_exc) - r_exc)
        r_inh += dt*((h_e2i - h_i2i + in_inh) - r_inh)

        np.maximum(r_exc, 0.0, r_exc)
        np.minimum(r_exc, f_max, r_exc)
        np.maximum(r_inh, 0.0, r_inh)
        np.minimum(r_inh, f_max, r_inh)

        if is_mean:
            r_hat[:] = r_exc.mean()
        else:
            r_hat = r_exc

        if is_hebbian:
            hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_i, c_e2i_j, r_inh, r_hat)
        else:
            non_hebbian(eta, rho0, J_max, c_e2i_data, c_e2i_j, r_hat)

        if not n % r_step:
            i = n / r_step
            rates[i, 0] = r_exc.mean()
            rates[i, 1] = np.power(r_exc, 2).mean()
            rates[i, 2] = r_exc.var()
            weights[i, :] = c_e2i_data.mean(), c_e2i_data.var()

    return rates, weights


def inner_dense(dt, eta, rho0, n_steps, NE, NI, r_exc, r_inh, in_exc, in_inh, f_max, J_max,
                       c_e2i_indptr, c_e2i_indices, c_e2i_data,
                       c_i2e, c_e2e, c_i2i, c_e2i_j, c_e2i_i, h_e2i, h_i2e, h_e2e, h_i2i,
                       r_step, rates, weights, is_hebbian, K=None):
    from numpy.fft import rfft, irfft
    from scipy.sparse.sparsetools import csr_matvec
    import numexpr as ne
    from numpy import take

    if K is not None:
        def get_rates(): return irfft(K * rfft(r_exc), NE).real
    else:
        def get_rates(): return r_exc
    r_exc_j = np.empty(c_e2i_j.size, dtype=dtype)
    r_inh_i = np.empty(c_e2i_i.size, dtype=dtype) if is_hebbian else 0.

    gd = dict(dt=dt, eta=eta, rho0=rho0, in_exc=in_exc, in_inh=in_inh, c_e2i_data=c_e2i_data,
              r_exc=r_exc, r_inh=r_inh, r_exc_j=r_exc_j, r_inh_i=r_inh_i)
    for n in xrange(n_steps):
        h_e2i.fill(0.)
        np.dot(c_i2i, r_inh, out=h_i2i)
        np.dot(c_i2e, r_inh, out=h_i2e)
        np.dot(c_e2e, r_exc, out=h_e2e)
        csr_matvec(NI, NE, c_e2i_indptr, c_e2i_indices, c_e2i_data, r_exc, h_e2i)

        ne.evaluate('r_exc + dt*((h_e2e - h_i2e + in_exc) - r_exc)', global_dict=gd, out=r_exc)
        np.maximum(r_exc, 0.0, r_exc)
        np.minimum(r_exc, f_max, r_exc)

        ne.evaluate('r_inh + dt*((h_e2i - h_i2i + in_inh) - r_inh)', global_dict=gd, out=r_inh)
        np.maximum(r_inh, 0.0, r_inh)
        np.minimum(r_inh, f_max, r_inh)

        take(get_rates(), c_e2i_j, out=r_exc_j)

        if is_hebbian:  # Hebbian
            take(r_inh, c_e2i_i, out=r_inh_i)
            ne.evaluate('c_e2i_data + eta * (r_exc_j - rho0) * r_inh_i', global_dict=gd, out=c_e2i_data)
        else:  # Non-Hebbian
            ne.evaluate('c_e2i_data + eta * (r_exc_j - rho0)', global_dict=gd, out=c_e2i_data)

        np.maximum(c_e2i_data, 0.0, c_e2i_data)
        np.minimum(c_e2i_data, J_max, c_e2i_data)

        if not n % r_step:
            i = n / r_step
            rates[i, 0] = r_exc.mean()
            rates[i, 1] = np.power(r_exc, 2).mean()
            rates[i, 2] = r_exc.var()
            weights[i, :] = c_e2i_data.mean(), c_e2i_data.var()

    return rates, weights


@jit(nopython=True, cache=cache)
def csr_matvec_jit(Nr, indptr, indices, data, r, h):
    for i in range(Nr):
        s = 0.
        for jj in range(indptr[i], indptr[i+1]):
            s += data[indices[jj]] * r[indices[jj]]
        h[i] = s


@jit(nopython=True, cache=cache)
def coo_matvec_jit(nnz, Ai, Aj, Ax, r, h):
    # 2-3 times slower than CSR because COO storage size is O(3 nnz) vs O(2 nnz + n) for CSR.
    for n in range(nnz):
        h[Ai[n]] += Ax[n] * r[Aj[n]]

import matplotlib.pyplot as plt
for s in np.logspace(3, 8., base=np.e, num=2):
    compute_connections_dense(s, .02, 1., 4000, 1., 4000, 1)
    compute_connections_dense(s, .02, 4., 1000, 4., 1000, 1)
    compute_connections_dense(s, .02, 1., 4000, 4., 1000, 1)
    compute_connections_dense(s, .02, 4., 1000, 1., 4000, 1)
plt.show()

def compute_connections_dense(p, N_pre, N_post, J):
    c = int(p * N_pre)
    row_ind = np.empty(c * N_post, dtype=np.int32)
    col_ind = np.empty(c * N_post, dtype=np.int32)
    conns = np.zeros((N_post, N_pre), dtype=dtype)
    for i in range(N_post):
        lb, ub = i*c, i*c+c
        j = np.random.choice(N_pre, c, replace=False)
        row_ind[lb:ub] = i
        col_ind[lb:ub] = j
        conns[i, j] = J  # np.random.choice(N_pre, c, replace=False)
    return conns, row_ind, col_ind
@jit(nopython=True, cache=cache)
def compute_coo(p, N_pre, N_post, J):
    """
    where ``data``, ``row_ind`` and ``col_ind`` satisfy the
            relationship ``a[row_ind[k], col_ind[k]] = data[k]``.
    """
    c = int(p * N_pre)
    row_ind = np.empty(c * N_post, dtype=np.int32)
    col_ind = np.empty(c * N_post, dtype=np.int32)
    for i in range(N_post):
        lb, ub = i*c, i*c+c
        row_ind[lb:ub] = i
        col_ind[lb:ub] = np.random.choice(N_pre, c, replace=False)
    data = np.ones(c * N_post, dtype=dtype) * J

    return row_ind, col_ind, data
'''
