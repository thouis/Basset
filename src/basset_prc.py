#!/usr/bin/env python

from optparse import OptionParser
import h5py
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve

################################################################################
# basset_prc.py
#
# Plot precision-recall curves.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <test_hdf5> <test_preds>'
    parser = OptionParser(usage)
    parser.add_option('-o', dest='out_dir', default='prc')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide test HDF5 and predictions')
    else:
        hdf5_file = args[0]
        preds_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #############################################
    # input targets and predictions
    #############################################
    h5_in = h5py.File(hdf5_file)
    targets = h5_in['test_out']
    if type(h5_in['target_labels'][0]) == np.bytes_:
        target_labels = np.array([label.decode('UTF-8') for label in h5_in['target_labels']])
    else:
        target_labels = np.array(h5_in['target_labels'])

    preds = np.genfromtxt(preds_file, delimiter='\t', dtype='float16')

    #############################################
    # make all PRC plots
    #############################################
    auc_out = open('%s/auc.txt' % options.out_dir, 'w')

    for ti in range(len(target_labels)):
        print('Target %d: %s' % (ti,target_labels[ti]))

        # compute AUC
        auc = average_precision_score(targets[:,ti], preds[:,ti])
        print('%d\t%s\t%f' % (ti, target_labels[ti], auc), file=auc_out)

        # compute curves
        prec, recall, thresh = precision_recall_curve(targets[:,ti], preds[:,ti])
        rate = sum(targets[:,ti]) / float(targets.shape[0])

        # print
        prc_out = open('%s/t%d.txt' % (options.out_dir, ti), 'w')
        for i in range(len(thresh)):
            print('%f %f %f' % (thresh[i], recall[i], prec[i]), file=prc_out)
        prc_out.close()

        # plot
        plt.figure(figsize=(6,6))

        plt.scatter(recall, prec, s=8, linewidths=0, c=sns.color_palette('deep')[0])
        plt.axhline(y=rate, c='black', linewidth=1, linestyle='--')

        plt.title(target_labels[ti])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.grid(True)
        plt.tight_layout()

        out_pdf = '%s/t%d.pdf' % (options.out_dir, ti)
        plt.savefig(out_pdf)
        plt.close()

    auc_out.close()

    h5_in.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
