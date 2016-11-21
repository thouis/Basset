#!/usr/bin/env python

from optparse import OptionParser
import os
from sklearn.metrics import log_loss
import subprocess
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import dna_io
from basset_profile import load_profile
from basset_sat import get_real_nt

'''
basset_refine.py

Refine a promising sequence to maximize its similarity with a desired activity profile.
'''

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <profile_file> <fasta_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='input_activity_file', help='Optional activity table corresponding to an input FASTA file')
    parser.add_option('-e', dest='norm_even', default=False, action='store_true', help='Normalize the weights for the positive and negative datasets to be even [Default: %default]')
    parser.add_option('--cuda', dest='cuda', default=False, action='store_true', help='Run on GPGPU [Default: %default]')
    parser.add_option('--cudnn', dest='cudnn', default=False, action='store_true', help='Run on GPGPU w/cuDNN [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='refine', help='Output directory [Default: %default]')
    parser.add_option('-s', dest='early_stop', default=.05, type='float', help='Proportion by which the mutation must improve to be accepted [Default: %default]')
    parser.add_option('-z', dest='weight_zero', default=1.0, type='float', help='Adjust the weights for the zero samples by this value [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide Basset model file, activity profile file, and sequence FASTA file')
    else:
        model_file = args[0]
        profile_file = args[1]
        input_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    gpgpu_str = ''
    if options.cudnn:
        gpgpu_str = '-cudnn'
    elif options.cuda:
        gpgpu_str = '-cuda'

    #################################################################
    # prep sequence
    #################################################################

    # load sequence
    seq = ''
    for line in open(input_file):
        if line[0] == '>':
            header = line[1:].rstrip()
        else:
            seq += line.rstrip()

    # convert to one hot coding
    seq_1hot = dna_io.dna_one_hot(seq)
    seq_1hot = np.reshape(seq_1hot, (1,4,1,len(seq)))

    # make initial predictions
    seq_preds = predict_seq(model_file, seq_1hot, gpgpu_str, options.out_dir)
    num_targets = seq_preds.shape[0]

    #################################################################
    # prep profile
    #################################################################
    activity_profile, profile_weights, profile_mask, target_labels = load_profile(profile_file, num_targets, options.norm_even, options.weight_zero)

    #################################################################
    # iteratively refine
    #################################################################
    nts = 'ACGT'
    local_max = False
    refined_profile_list = [seq_preds[profile_mask]]
    ri = 1
    while not local_max:
        print('Refinement stage %d' % ri)

        # write sequence to HDF5
        seq_hdf5_file = '%s/seq%d.h5' % (options.out_dir,ri)
        seq_hdf5_out = h5py.File(seq_hdf5_file, 'w')
        seq_hdf5_out.create_dataset('test_in', data=seq_1hot)
        seq_hdf5_out.close()

        # perform saturated mutagenesis
        sat_hdf5_file = '%s/satmut%d.h5' % (options.out_dir,ri)
        torch_cmd = 'basset_sat_predict.lua %s -rc %s %s %s' % (gpgpu_str, model_file, seq_hdf5_file, sat_hdf5_file)
        subprocess.call(torch_cmd, shell=True)

        # read results into 4 x L x T
        sat_hdf5_in = h5py.File(sat_hdf5_file, 'r')
        seq_mod_preds = np.array(sat_hdf5_in['seq_mod_preds'])
        seq_mod_preds = seq_mod_preds.squeeze()
        sat_hdf5_in.close()

        # find sequence prediction
        ni, li = get_real_nt(seq)
        seq_pred = seq_mod_preds[ni,li,:]

        # set to min
        seq_dist = log_loss(activity_profile[profile_mask], seq_mod_preds[ni,li,profile_mask], sample_weight=profile_weights[profile_mask])
        min_dist = seq_dist
        min_entry = (li,ni)
        local_max = True

        # consider mutated sequences
        for li in range(len(seq)):
            for ni in range(4):
                if seq_1hot[0,ni,0,li] == 0:
                    # compute distance
                    mut_dist = log_loss(activity_profile[profile_mask], seq_mod_preds[ni,li,profile_mask], sample_weight=profile_weights[profile_mask])

                    # compare to min
                    if mut_dist*1.05 < min_dist:
                        local_max = False
                        min_dist = mut_dist
                        min_entry = (li,ni)

        # update
        if local_max:
            print(' Maximized')
        else:
            # update trace
            li, ni = min_entry
            print(' Mutate %d %s --> %s' % (li, seq[li], nts[ni]))
            print(' Distance decreases from %.3f to %.3f' % (seq_dist, min_dist))

            # update sequence
            seq = seq[:li] + nts[ni] + seq[li+1:]
            dna_io.one_hot_set(seq_1hot[0], li, nts[ni])

            # save profile
            refined_profile_list.append(seq_mod_preds[ni,li,profile_mask])

        ri += 1

    #################################################################
    # finish
    #################################################################
    refined_profiles = np.array(refined_profile_list)

    # print refinement table
    table_out = open('%s/final_table.txt' % options.out_dir, 'w')
    for ri in range(refined_profiles.shape[0]):
        pi = 0
        for ti in range(num_targets):
            if profile_mask[ti]:
                cols = (ri, ti, refined_profiles[ri,pi])
                print('%-3d  %3d  %.3f' % cols, file=table_out)
                pi += 1
    table_out.close()


    # heat map
    if len(refined_profile_list) > 1:
        plt.figure()
        g = sns.clustermap(np.transpose(refined_profiles), col_cluster=False, metric='euclidean', linewidths=0, yticklabels=target_labels[profile_mask], xticklabels=False)
        plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
        plt.savefig('%s/final_heat.pdf' % options.out_dir)
        plt.close()

    # output sequence
    final_fasta_file = '%s/final_seq.fa' % options.out_dir
    final_fasta_out = open(final_fasta_file, 'w')
    print('>%s\n%s' % (header, seq), file=final_fasta_out)
    final_fasta_out.close()

    # perform a new saturated mutagenesis
    satmut_targets = ','.join([str(ti) for ti in range(len(activity_profile)) if profile_mask[ti]])
    if gpgpu_str != '':
        gpgpu_str = '-%s' % gpgpu_str
    cmd = 'basset_sat.py %s -n 500 -o %s/final_satmut -t %s %s %s' % (gpgpu_str, options.out_dir, satmut_targets, model_file, final_fasta_file)
    subprocess.call(cmd, shell=True)


def predict_seq(model_file, seq_1hot, gpgpu_str, out_dir):
    ''' Make predictions for the single input sequence. '''

    # write sequence to HDF5
    seq_hdf5_file = '%s/seq0.h5' % out_dir
    seq_hdf5_out = h5py.File(seq_hdf5_file, 'w')
    seq_hdf5_out.create_dataset('test_in', data=seq_1hot)
    seq_hdf5_out.close()

    # predict
    preds_file = '%s/preds0.txt' % out_dir
    torch_cmd = 'basset_predict.lua %s %s %s %s' % (gpgpu_str, model_file, seq_hdf5_file, preds_file)
    subprocess.call(torch_cmd, shell=True)

    # read predictions
    seq_preds = np.loadtxt(preds_file)

    return seq_preds


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
