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

'''
basset_profile.py

Measure the similarity between test sequences and a desired activity profile.

Notes:
 -To examine all sequences, provide the FASTA as input and targets as -a.
'''

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <profile_file> <input_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='input_activity_file', help='Optional activity table corresponding to an input FASTA file')
    parser.add_option('--cuda', dest='cuda', default=False, action='store_true', help='Run on GPGPU [Default: %default]')
    parser.add_option('--cudnn', dest='cudnn', default=False, action='store_true', help='Run on GPGPU w/cuDNN [Default: %default]')
    parser.add_option('-d', dest='model_out_file', default=None, help='Pre-computed model predictions output table [Default: %default]')
    parser.add_option('-e', dest='norm_even', default=False, action='store_true', help='Normalize the weights for the positive and negative datasets to be even [Default: %default]')
    parser.add_option('-f', dest='font_heat', default=6, type='int', help='Heat map axis font size [Default: %default]')
    parser.add_option('-n', dest='num_dissect', default=10, type='int', help='Dissect the top n hits [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='profile', help='Output directory [Default: %default]')
    parser.add_option('-z', dest='weight_zero', default=1.0, type='float', help='Adjust the weights for the zero samples by this value [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide Basset model file, activity profile file, and input sequences (as a FASTA file or test data in an HDF file)')
    else:
        model_file = args[0]
        profile_file = args[1]
        input_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # parse input file
    #################################################################
    try:
        # input_file is FASTA

        # load sequences and headers
        seqs = []
        seq_headers = []
        for line in open(input_file):
            if line[0] == '>':
                seq_headers.append(line[1:].rstrip())
                seqs.append('')
            else:
                seqs[-1] += line.rstrip()

        # convert to arrays
        seqs = np.array(seqs)
        seq_headers = np.array(seq_headers)

        model_input_hdf5 = '%s/model_in.h5'%options.out_dir

        if options.input_activity_file:
            # one hot code
            seqs_1hot, targets = dna_io.load_data_1hot(input_file, options.input_activity_file, mean_norm=False, whiten=False, permute=False, sort=False)

        else:
            # load sequences
            seqs_1hot = dna_io.load_sequences(input_file, permute=False)
            targets = None

        # reshape sequences for torch
        seqs_1hot = seqs_1hot.reshape((seqs_1hot.shape[0],4,1,seqs_1hot.shape[1]/4))

        # write as test data to a HDF5 file
        h5f = h5py.File(model_input_hdf5, 'w')
        h5f.create_dataset('test_in', data=seqs_1hot)
        h5f.close()

    except (IOError, IndexError, UnicodeDecodeError):
        # input_file is HDF5

        try:
            model_input_hdf5 = input_file

            # load (sampled) test data from HDF5
            hdf5_in = h5py.File(input_file, 'r')
            seqs_1hot = np.array(hdf5_in['test_in'])
            targets = np.array(hdf5_in['test_out'])
            try: # TEMP
                seq_headers = np.array([h.decode('UTF-8') for h in hdf5_in['test_headers']])
                # seq_headers = np.array(hdf5_in['test_headers'])
            except:
                seq_headers = None
            hdf5_in.close()

            # convert to ACGT sequences
            seqs = dna_io.vecs2dna(seqs_1hot)

        except IOError:
            parser.error('Could not parse input file as FASTA or HDF5.')


    #################################################################
    # Torch predict modifications
    #################################################################
    # GPU options (needed below, too)
    gpgpu_str = ''
    if options.cudnn:
        gpgpu_str = '-cudnn'
    elif options.cuda:
        gpgpu_str = '-cuda'

    if options.model_out_file is None:
        options.model_out_file = '%s/preds.txt' % options.out_dir

        torch_cmd = 'basset_predict.lua -rc %s %s %s %s' % (gpgpu_str, model_file, model_input_hdf5, options.model_out_file)
        print(torch_cmd)
        subprocess.call(torch_cmd, shell=True)

    # read in predictions
    seqs_preds = np.loadtxt(options.model_out_file)

    num_targets = seqs_preds.shape[1]


    #################################################################
    # parse profile file
    #################################################################
    activity_profile, profile_weights, profile_mask, target_labels = load_profile(profile_file, num_targets, options.norm_even, options.weight_zero)

    #################################################################
    # plot clustered heat map limited to relevant targets
    #################################################################
    seqs_preds_prof = seqs_preds[:,profile_mask]
    seqs_preds_var = seqs_preds_prof.var(axis=1)
    seqs_sort_var = np.argsort(seqs_preds_var)[::-1]

    plt.figure()
    g = sns.clustermap(np.transpose(seqs_preds_prof[seqs_sort_var[:1000]]), metric='euclidean', linewidths=0, yticklabels=target_labels[profile_mask], xticklabels=False)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    for label in g.ax_heatmap.yaxis.get_majorticklabels():
        label.set_fontsize(options.font_heat)
    plt.savefig('%s/heat_clust.pdf' % options.out_dir)
    plt.close()


    #################################################################
    # compute profile distances
    #################################################################
    # compute prediction distances
    seqs_pdists = []
    for si in range(seqs_preds.shape[0]):
        # sd = np.power(seqs_preds[si,profile_mask]-activity_profile[profile_mask], 2).sum()
        sd = log_loss(activity_profile[profile_mask], seqs_preds[si,profile_mask], sample_weight=profile_weights[profile_mask])
        seqs_pdists.append(sd)
    seqs_pdists = np.array(seqs_pdists)

    # obtain sorted indexes
    seqs_sort_dist = np.argsort(seqs_pdists)

    # compute target distances
    seqs_tdists = []
    for si in range(seqs_preds.shape[0]):
        tdists = np.absolute(targets[si,profile_mask]-activity_profile[profile_mask])
        tdists_weight = np.multiply(tdists, profile_weights[profile_mask])
        td = tdists_weight.sum()
        seqs_tdists.append(td)
    seqs_tdists = np.array(seqs_tdists)

    # print as table
    table_out = open('%s/table.txt' % options.out_dir, 'w')
    for si in seqs_sort_dist:
        cols = [si, seqs_pdists[si], seqs_tdists[si]] + list(seqs_preds[si,profile_mask])
        print('\t'.join([str(c) for c in cols]), file=table_out)
    table_out.close()


    #################################################################
    # plot sorted heat map
    #################################################################
    plt.figure()
    g = sns.clustermap(np.transpose(seqs_preds_prof[seqs_sort_dist[:1000]]), col_cluster=False, metric='euclidean', linewidths=0, yticklabels=target_labels[profile_mask], xticklabels=False)
    plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    for label in ax.yaxis.get_majorticklabels():
        label.set_fontsize(options.font_heat)
    plt.savefig('%s/heat_rank.pdf' % options.out_dir)
    plt.close()


    #################################################################
    # dissect the top hits
    #################################################################
    satmut_targets = ','.join([str(ti) for ti in range(len(activity_profile)) if profile_mask[ti]])

    for ni in range(options.num_dissect):
        si = seqs_sort_dist[ni]

        # print FASTA
        fasta_file = '%s/seq%d.fa' % (options.out_dir, ni)
        fasta_out = open(fasta_file, 'w')
        print('>%s\n%s' % (seq_headers[si],seqs[si]), file=fasta_out)
        fasta_out.close()

        # saturated mutagenesis
        if gpgpu_str != '':
            gpgpu_str = '-%s' % gpgpu_str
        cmd = 'basset_sat.py %s -n 500 -o %s/satmut%d -t %s %s %s' % (gpgpu_str, options.out_dir, ni, satmut_targets, model_file, fasta_file)
        subprocess.call(cmd, shell=True)

        # predictions and targets heat
        profile_sort = np.argsort(activity_profile[profile_mask])
        heat_mat = np.array([activity_profile[profile_mask], targets[si,profile_mask], seqs_preds_prof[si]])
        heat_mat = heat_mat[:,profile_sort]

        plt.figure()
        ax = sns.heatmap(np.transpose(heat_mat), yticklabels=target_labels[profile_mask][profile_sort], xticklabels=['Desired', 'Experiment', 'Prediction'])
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45)
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=-0)
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(options.font_heat)
        plt.savefig('%s/heat%d.pdf' % (options.out_dir,ni))
        plt.close()


def load_profile(profile_file, num_targets, norm_even=False, weight_zero=1):
    ''' Load the desired profile from file. '''

    # read from file
    activity_profile = []
    profile_weights = []
    target_labels = []
    for line in open(profile_file):
        a = line.split()
        ti = int(a[0])
        ta = float(a[1])
        if len(a) > 2:
            tw = float(a[2])
        else:
            tw = 1
        if len(a) > 3:
            tlabel = a[3]

        while len(activity_profile) < ti:
            activity_profile.append(np.nan)
            profile_weights.append(0)
            target_labels.append(None)

        activity_profile.append(ta)
        profile_weights.append(tw)
        target_labels.append(tlabel)

    while len(activity_profile) < num_targets:
        activity_profile.append(np.nan)
        profile_weights.append(0)
        target_labels.append(None)

    # convert to array
    activity_profile = np.array(activity_profile)
    profile_weights = np.array(profile_weights)
    target_labels = np.array(target_labels)

    # compute mask
    profile_mask = np.logical_not(np.isnan(activity_profile))

    # normalize positives versus zeros
    if norm_even:
        # compute weight sums
        sum_on = 0
        sum_off = 0
        for ti in range(activity_profile.shape[0]):
            if profile_mask[ti]:
                if activity_profile[ti] > 0:
                    sum_on += profile_weights[ti]
                else:
                    sum_off += profile_weights[ti]

        # adjust weights
        for ti in range(activity_profile.shape[0]):
            if profile_mask[ti]:
                if activity_profile[ti] > 0:
                    profile_weights[ti] /= sum_on
                else:
                    profile_weights[ti] /= sum_off

    # up-weight zero's
    if weight_zero:
        for ti in range(activity_profile.shape[0]):
            if profile_mask[ti] and activity_profile[ti] == 0:
                profile_weights[ti] *= weight_zero

    return activity_profile, profile_weights, profile_mask, target_labels


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
