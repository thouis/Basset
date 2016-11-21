#!/usr/bin/env python

from optparse import OptionParser
import pdb, os, subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dna_io import dna_one_hot
import vcf

################################################################################
# basset_sad.py
#
# Compute SNP Accessibility Difference scores for SNPs in a VCF file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_th> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='csv', default=False, action='store_true', help='Print table as CSV [Default: %default]')
    parser.add_option('--cuda', dest='cuda', default=False, action='store_true', help='Predict on the GPU [Default: %default]')
    parser.add_option('--cudnn', dest='cudnn', default=False, action='store_true', help='Predict on the GPU w/ CuDNN [Default: %default]')
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5 [Default: %default]')
    parser.add_option('--dense', dest='dense_table', default=False, action='store_true', help='Print a dense SNP x Targets table, as opposed to a SNP/Target pair per line [Default: %default]')
    parser.add_option('-e', dest='heatmaps', default=False, action='store_true', help='Draw score heatmaps, grouped by index SNP [Default: %default]')
    parser.add_option('-f', dest='genome_fasta', default='%s/data/genomes/hg19.fa'%os.environ['BASSETDIR'], help='Genome FASTA from which sequences will be drawn [Default: %default]')
    parser.add_option('--f1', dest='genome1_fasta', default=None, help='Genome FASTA which which major allele sequences will be drawn')
    parser.add_option('--f2', dest='genome2_fasta', default=None, help='Genome FASTA which which minor allele sequences will be drawn')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=1000, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-m', dest='min_limit', default=0.1, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='File specifying target indexes and labels in table format')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Torch model and VCF file')
    else:
        model_th = args[0]
        vcf_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # prep SNP sequences
    #################################################################
    # load SNPs
    snps = vcf.vcf_snps(vcf_file, options.index_snp, options.score, options.genome2_fasta is not None)

    # get one hot coded input sequences
    if not options.genome1_fasta or not options.genome2_fasta:
        seq_vecs, seq_headers, snps = vcf.snps_seq1(snps, options.seq_len, options.genome_fasta)
    else:
        seq_vecs, seq_headers, snps = vcf.snps2_seq1(snps, options.seq_len, options.genome1_fasta, options.genome2_fasta)

    # reshape sequences for torch
    seq_vecs = seq_vecs.reshape((seq_vecs.shape[0],4,1,seq_vecs.shape[1]//4))

    # write to HDF5
    h5f = h5py.File('%s/model_in.h5'%options.out_dir, 'w')
    h5f.create_dataset('test_in', data=seq_vecs)
    h5f.close()


    #################################################################
    # predict in Torch
    #################################################################
    if options.model_hdf5_file is None:
        if options.cudnn:
            cuda_str = '-cudnn'
        elif options.cuda:
            cuda_str = '-cuda'
        else:
            cuda_str = ''

        options.model_hdf5_file = '%s/model_out.txt' % options.out_dir
        cmd = 'basset_predict.lua -rc %s %s %s/model_in.h5 %s' % (cuda_str, model_th, options.out_dir, options.model_hdf5_file)
        print(cmd)
        subprocess.call(cmd, shell=True)

    # read in predictions
    seq_preds = []
    for line in open(options.model_hdf5_file):
        seq_preds.append(np.array([np.float16(p) for p in line.split()]))
    seq_preds = np.array(seq_preds)


    #################################################################
    # collect and print SADs
    #################################################################
    if options.targets_file is None:
        target_labels = ['t%d' % ti for ti in range(seq_preds.shape[1])]
    else:
        target_labels = [line.split()[0] for line in open(options.targets_file)]

    if options.dense_table:
        sad_out = open('%s/sad_table.txt' % options.out_dir, 'w')
    else:
        header_cols = ('rsid', 'index', 'score', 'ref', 'alt', 'target', 'ref_pred', 'alt pred', 'sad')
        if options.csv:
            sad_out = open('%s/sad_table.csv' % options.out_dir, 'w')
            print(','.join(header_cols), file=sad_out)
        else:
            sad_out = open('%s/sad_table.txt' % options.out_dir, 'w')
            print(' '.join(header_cols), file=sad_out)

    # hash by index snp
    sad_matrices = {}
    sad_labels = {}
    sad_scores = {}

    pi = 0
    for snp in snps:
        # get reference prediction
        ref_preds = seq_preds[pi,:]
        pi += 1

        for alt_al in snp.alt_alleles:
            # get alternate prediction
            alt_preds = seq_preds[pi,:]
            pi += 1

            # normalize by reference
            alt_sad = alt_preds - ref_preds
            sad_matrices.setdefault(snp.index_snp,[]).append(alt_sad)

            # label as mutation from reference
            alt_label = '%s_%s>%s' % (snp.rsid, vcf.cap_allele(snp.ref_allele), vcf.cap_allele(alt_al))
            sad_labels.setdefault(snp.index_snp,[]).append(alt_label)

            # save scores
            sad_scores.setdefault(snp.index_snp,[]).append(snp.score)

            # set index SNP
            snp_is = '%-13s' % '.'
            if options.index_snp:
                snp_is = '%-13s' % snp.index_snp

            # set score
            snp_score = '%5s' % '.'
            if options.score:
                snp_score = '%5.3f' % snp.score

            # print table line(s)
            if options.dense_table:
                cols = [snp.rsid, snp_is, snp_score, vcf.cap_allele(snp.ref_allele), vcf.cap_allele(alt_al)]
                for ti in range(len(alt_sad)):
                    cols += ['%.4f'%ref_preds[ti], '%.4f'%alt_sad[ti]]

                sep = ' '
                if options.csv:
                    sep = ','

                print(sep.join([str(c) for c in cols]), file=sad_out)

            else:
                for ti in range(len(alt_sad)):
                    cols = (snp.rsid, snp_is, snp_score, vcf.cap_allele(snp.ref_allele), vcf.cap_allele(alt_al), target_labels[ti], ref_preds[ti], alt_preds[ti], alt_sad[ti])
                    if options.csv:
                        print(','.join([str(c) for c in cols]), file=sad_out)
                    else:
                        print('%-13s %s %5s %6s %6s %12s %6.4f %6.4f %7.4f' % cols, file=sad_out)

    sad_out.close()


    #################################################################
    # plot SAD heatmaps
    #################################################################
    if options.heatmaps:
        for ii in sad_matrices:
            # convert fully to numpy arrays
            sad_matrix = abs(np.array(sad_matrices[ii]))
            print(ii, sad_matrix.shape)

            if sad_matrix.shape[0] > 1:
                vlim = max(options.min_limit, sad_matrix.max())
                score_mat = np.reshape(np.array(sad_scores[ii]), (-1, 1))

                # plot heatmap
                plt.figure(figsize=(20, 0.5 + 0.5*sad_matrix.shape[0]))

                if options.score:
                    # lay out scores
                    cols = 12
                    ax_score = plt.subplot2grid((1,cols), (0,0))
                    ax_sad = plt.subplot2grid((1,cols), (0,1), colspan=(cols-1))

                    sns.heatmap(score_mat, xticklabels=False, yticklabels=False, vmin=0, vmax=1, cmap='Reds', cbar=False, ax=ax_score)
                else:
                    ax_sad = plt.gca()

                sns.heatmap(sad_matrix, xticklabels=target_labels, yticklabels=sad_labels[ii], vmin=0, vmax=vlim, ax=ax_sad)

                for tick in ax_sad.get_xticklabels():
                    tick.set_rotation(-45)
                    tick.set_horizontalalignment('left')
                    tick.set_fontsize(5)

                plt.tight_layout()
                if ii == '.':
                    out_pdf = '%s/sad_heat.pdf' % options.out_dir
                else:
                    out_pdf = '%s/sad_%s_heat.pdf' % (options.out_dir, ii)
                plt.savefig(out_pdf)
                plt.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)