import sys
import h5py
import numpy as np
from scipy.signal import correlate2d
from dendro import show_dendro
import pylab


def padcrosscorr(filt1, filt2, l):
    assert filt1.shape[0] == l
    assert filt2.shape[0] == l
    assert filt1.shape[1] == filt2.shape[1] == 4
    # pad along sequence, but keep other dim (== 4) the same
    filt1 = np.pad(filt1, (l - 1, 0), 'constant')
    filt2 = np.pad(filt2, (l - 1, 0), 'constant')
    return correlate2d(filt1, filt2, 'valid')


def similarities(w1, w2, l, n):
    sims = np.zeros((n, n))
    for idx1 in range(n):
        for idx2 in range(n):
            sims[idx1, idx2] = padcrosscorr(w1[:, 0, :, idx1], w2[:, 0, :, idx2], l).max()
    return sims

if __name__ == '__main__':
    f1 = h5py.File(sys.argv[1], 'r')
    f2 = h5py.File(sys.argv[2], 'r')

    w1 = f1['convolution1d_1']['convolution1d_1_W'][...]
    w2 = f2['convolution1d_1']['convolution1d_1_W'][...]
    l, o, fr, n = w1.shape
    assert o == 1
    assert fr == 4

    # we subtract off the minimum at each location, as it's effectively a bias
    # and doing so converts the location to a weighted probability distribution
    w1 = w1 - np.min(w1, axis=2, keepdims=True)
    w2 = w2 - np.min(w2, axis=2, keepdims=True)
    print (w1.shape, w1.max())

    cross_sim = similarities(w1, w2, l, n)
    assert cross_sim.min() >= 0.0
    selfsim_1 = similarities(w1, w1, l, n)
    selfsim_2 = similarities(w2, w2, l, n)

    norms_1 = 1.0 / np.sqrt(np.diag(selfsim_1)).reshape((-1, 1))
    norms_2 = 1.0 / np.sqrt(np.diag(selfsim_2)).reshape((1, -1))
    assert np.isfinite(norms_1).all()
    assert np.isfinite(norms_2).all()

    cross_sim = (cross_sim * norms_1) * norms_2
    selfsim_1 = (selfsim_1 * norms_1) * norms_1.T
    selfsim_t = (selfsim_2 * norms_2.T) * norms_2
    selfsim_1 = (selfsim_1 + selfsim_1.T) / 2.0
    selfsim_2 = (selfsim_2 + selfsim_2.T) / 2.0

    dists_1 = 1 - selfsim_1
    dists_2 = 1 - selfsim_2
    np.fill_diagonal(dists_1, 0)
    np.fill_diagonal(dists_2, 0)

    print(";lot")
    show_dendro(cross_sim, dists_1, dists_2)
    pylab.show()
