import os
import sys
import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi

raw_files = sorted(glob('data/grayscale_maps/*.png'))
# ignore first file because it's not labeled, and move to grayscale
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)[1:, ..., 0]
print(raw.shape)

gt = h5py.File('data/seg_groundtruth.h5', 'r')['main'][1:, ...]
print(gt.shape)

membranes = gt.copy()
for idx in range(membranes.shape[0]):
    mem2d = membranes[idx, ...]
    borders = ndi.minimum_filter(mem2d, 3) != ndi.maximum_filter(mem2d, 3)
    borders = borders | (mem2d == 0)
    membranes[idx, ...] = borders

f = h5py.File('training_data.h5', 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)
f.create_dataset('membranes', data=membranes.astype(np.uint8))
