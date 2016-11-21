#!/usr/bin/env python
from optparse import OptionParser
import copy, os, pdb, subprocess, sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

import basset_sat
import dna_io
from seq_logo import seq_logo
import vcf

################################################################################
# basset_sat_vcf.py
#
# Perform an in silico saturated mutagenesis of the regions surrounding a list
# of SNPs given in VCF format using the given model.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5 [Default: %default]')
    parser.add_option('-f', dest='genome_fasta', default='%s/data/genomes/hg19.fa'%os.environ['BASSETDIR'], help='Genome FASTA from which sequences will be drawn [Default: %default]')
    parser.add_option('--f1', dest='genome1_fasta', default=None, help='Genome FASTA which which major allele sequences will be drawn')
    parser.add_option('--f2', dest='genome2_fasta', default=None, help='Genome FASTA which which minor allele sequences will be drawn')
    parser.add_option('-g', dest='gain_height', default=False, action='store_true', help='Nucleotide heights determined by the max of loss and gain [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-m', dest='min_limit', default=0.1, type='float', help='Minimum heatmap limit [Default: %default]')
    parser.add_option('-n', dest='center_nt', default=200, type='int', help='Nt around the SNP to mutate and plot in the heatmap [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-t', dest='targets', default='-1', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and input SNPs in VCF format')
    else:
        model_file = args[0]
        vcf_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # prep SNP sequences
    #################################################################
    # load SNPs
    snps = vcf.vcf_snps(vcf_file, pos2=(options.genome2_fasta is not None))

    # get one hot coded input sequences
    if not options.genome1_fasta or not options.genome2_fasta:
        seqs_1hot, seq_headers, snps, seqs = vcf.snps_seq1(snps, options.seq_len, options.genome_fasta, return_seqs=True)
    else:
        seqs_1hot, seq_headers, snps, seqs = vcf.snps2_seq1(snps, options.seq_len, options.genome1_fasta, options.genome2_fasta, return_seqs=True)

    # reshape sequences for torch
    seqs_1hot = seqs_1hot.reshape((seqs_1hot.shape[0],4,1,seqs_1hot.shape[1]/4))

    # write to HDF5
    model_input_hdf5 = '%s/model_in.h5'%options.out_dir
    h5f = h5py.File(model_input_hdf5, 'w')
    h5f.create_dataset('test_in', data=seqs_1hot)
    h5f.close()


    #################################################################
    # Torch predict modifications
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_sat_predict.lua -center_nt %d %s %s %s' % (options.center_nt, model_file, model_input_hdf5, options.model_hdf5_file)
        subprocess.call(torch_cmd, shell=True)


    #################################################################
    # load modification predictions
    #################################################################
    hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    seq_mod_preds = np.array(hdf5_in['seq_mod_preds'])
    hdf5_in.close()

    # trim seqs to match seq_mod_preds length
    delta_start = 0
    delta_len = seq_mod_preds.shape[2]
    if delta_len < options.seq_len:
        delta_start = (options.seq_len - delta_len)/2
        for si in range(len(seqs)):
            seqs[si] = seqs[si][delta_start:delta_start+delta_len]

    # decide which cells to plot
    if options.targets == '-1':
        plot_cells = range(seq_mod_preds.shape[3])
    else:
        plot_cells = [int(ci) for ci in options.targets.split(',')]


    #################################################################
    # plot
    #################################################################
    table_out = open('%s/table.txt' % options.out_dir, 'w')

    rdbu = sns.color_palette("RdBu_r", 10)

    nts = 'ACGT'
    for si in range(seq_mod_preds.shape[0]):
        header = seq_headers[si]
        seq = seqs[si]

        # plot some descriptive heatmaps for each individual cell type
        for ci in plot_cells:
            seq_mod_preds_cell = seq_mod_preds[si,:,:,ci]
            real_pred_cell = basset_sat.get_real_pred(seq_mod_preds_cell, seq)

            # compute matrices
            norm_matrix = seq_mod_preds_cell - real_pred_cell
            min_scores = seq_mod_preds_cell.min(axis=0)
            max_scores = seq_mod_preds_cell.max(axis=0)
            minmax_matrix = np.vstack([min_scores - real_pred_cell, max_scores - real_pred_cell])

            # prepare figure
            sns.set(style='white', font_scale=0.5)
            sns.axes_style({'axes.linewidth':1})
            spp = basset_sat.subplot_params(seq_mod_preds_cell.shape[1])
            fig = plt.figure(figsize=(20,3))
            ax_logo = plt.subplot2grid((3,spp['heat_cols']), (0,spp['logo_start']), colspan=(spp['logo_end']-spp['logo_start']))
            ax_sad = plt.subplot2grid((3,spp['heat_cols']), (1,spp['sad_start']), colspan=(spp['sad_end']-spp['sad_start']))
            ax_heat = plt.subplot2grid((3,spp['heat_cols']), (2,0), colspan=spp['heat_cols'])

            # print a WebLogo of the sequence
            vlim = max(options.min_limit, abs(minmax_matrix).max())
            if options.gain_height:
                seq_heights = 0.25 + 1.75/vlim*(abs(minmax_matrix).max(axis=0))
            else:
                seq_heights = 0.25 + 1.75/vlim*(-minmax_matrix[0])
            logo_eps = '%s/%s_c%d_seq.eps' % (options.out_dir, header.replace(':','_'), ci)
            seq_logo(seq, seq_heights, logo_eps, color_mode='meme')

            # add to figure
            logo_png = '%s.png' % logo_eps[:-4]
            subprocess.call('convert -density 300 %s %s' % (logo_eps, logo_png), shell=True)
            logo = Image.open(logo_png)
            ax_logo.imshow(logo)
            ax_logo.set_axis_off()

            # plot loss and gain SAD scores
            ax_sad.plot(-minmax_matrix[0], c=rdbu[0], label='loss', linewidth=1)
            ax_sad.plot(minmax_matrix[1], c=rdbu[-1], label='gain', linewidth=1)
            ax_sad.set_xlim(0,minmax_matrix.shape[1])
            ax_sad.legend()
            # ax_sad.grid(True, linestyle=':')
            for axis in ['top','bottom','left','right']:
                ax_sad.spines[axis].set_linewidth(0.5)

            # plot real-normalized scores
            vlim = max(options.min_limit, abs(norm_matrix).max())
            sns.heatmap(norm_matrix, linewidths=0, cmap='RdBu_r', vmin=-vlim, vmax=vlim, xticklabels=False, ax=ax_heat)
            ax_heat.yaxis.set_ticklabels('TGCA', rotation='horizontal') # , size=10)

            # save final figure
            plt.tight_layout()
            plt.savefig('%s/%s_c%d_heat.pdf' % (options.out_dir,header.replace(':','_'), ci), dpi=300)
            plt.close()


        #################################################################
        # print table of nt variability for each cell
        #################################################################
        for ci in range(seq_mod_preds.shape[3]):
            seq_mod_preds_cell = seq_mod_preds[si,:,:,ci]
            real_pred_cell = basset_sat.get_real_pred(seq_mod_preds_cell, seq)

            min_scores = seq_mod_preds_cell.min(axis=0)
            max_scores = seq_mod_preds_cell.max(axis=0)

            loss_matrix = real_pred_cell - seq_mod_preds_cell.min(axis=0)
            gain_matrix = seq_mod_preds_cell.max(axis=0) - real_pred_cell

            for pos in range(seq_mod_preds_cell.shape[1]):
                cols = [header, delta_start+pos, ci, loss_matrix[pos], gain_matrix[pos]]
                print('\t'.join([str(c) for c in cols]), file=table_out)

    table_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
