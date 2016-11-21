#!/usr/bin/env python

from optparse import OptionParser
import os
import random
import subprocess

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import pysam
from scipy.stats.mstats import mquantiles
import seaborn as sns

import stats

################################################################################
# basset_sick_gain.py
#
# Shuffle SNPs outside of DNase sites and compare the SAD distributions.
#
# Todo:
#  -Properly handle indels.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <vcf_file> <excl_bed_file> <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='cuda', default=False, action='store_true', help='Run on GPU [Default: %default]')
    parser.add_option('-e', dest='add_excl_bed', default='%s/assembly/hg19_gaps.bed'%os.environ['HG19'], help='Additional genomic regions to exclude from the shuffle [Default: %default]')
    parser.add_option('-f', dest='genome_fasta', default='%s/assembly/hg19.fa'%os.environ['HG19'], help='Genome FASTA [Default: %default]')
    parser.add_option('-g', dest='genome_file', default='%s/assembly/human.hg19.core.genome'%os.environ['HG19'], help='Genome file for shuffling [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad_shuffle', help='Output directory')
    parser.add_option('-r', dest='replot', default=False, action='store_true', help='Re-plot only, without re-computing [Default: %default]')
    parser.add_option('-s', dest='num_shuffles', default=1, type='int', help='Number of SNP shuffles [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='Target index, sample name table for targets to plot [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide VCF file, excluded BED file, and model file')
    else:
        vcf_file = args[0]
        excl_bed_file = args[1]
        model_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #########################################
    # supplement the excluded sites
    #########################################
    if options.add_excl_bed is not None:
        supp_excl_bed_file = '%s/excl.bed' % options.out_dir
        supp_excl_bed_out = open(supp_excl_bed_file, 'w')

        # copy exclusion BED file
        for line in open(excl_bed_file):
            a = line.split()
            print('\t'.join(a[:3]), file=supp_excl_bed_out)

        # add on additional sites
        for line in open(options.add_excl_bed):
            a = line.split()
            print('\t'.join(a[:3]), file=supp_excl_bed_out)

        supp_excl_bed_out.close()
        excl_bed_file = supp_excl_bed_file

    #########################################
    # compute SAD
    #########################################
    # filter VCF to excluded SNPs
    excl_vcf_file = '%s/excl.vcf' % options.out_dir
    if not options.replot:
        exclude_vcf(vcf_file, excl_bed_file, excl_vcf_file)

    # compute SADs
    true_sad = compute_sad(excl_vcf_file, model_file, '%s/excl_sad'%options.out_dir, options.seq_len, options.cuda, options.replot)

    #########################################
    # compute shuffled SAD
    #########################################
    # open reference genome
    genome_open = pysam.Fastafile(options.genome_fasta)

    shuffle_sad = np.zeros((true_sad.shape[0],true_sad.shape[1],options.num_shuffles))
    for ni in range(options.num_shuffles):
        # shuffle the SNPs
        shuf_vcf_file = '%s/shuf%d.vcf' % (options.out_dir, ni)
        shuffle_snps(excl_vcf_file, shuf_vcf_file, excl_bed_file, options.genome_file, genome_open)

        # compute SAD scores for shuffled SNPs
        shuffle_sad[:,:,ni] = compute_sad(shuf_vcf_file, model_file, '%s/shuf%d_sad'%(options.out_dir,ni), options.seq_len, options.cuda, options.replot)

    # compute shuffle means
    shuffle_sad_mean = shuffle_sad.mean(axis=2)

    #########################################
    # stats and plots
    #########################################
    targets = {}
    if options.targets_file:
        for line in open(options.targets_file):
            a = line.split()
            targets[int(a[0])] = a[1]
    else:
        for ti in range(true_sad.shape[1]):
            targets[ti] = 't%d' % ti

    mw_out = open('%s/mannwhitney.txt' % options.out_dir, 'w')

    # plot defaults
    sns.set(font_scale=1.5, style='ticks')

    for ti in targets:
        # plot CDFs
        sns_colors = sns.color_palette('deep')
        plt.figure()
        plt.hist(true_sad[:,ti], 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[0], linewidth=1, label='SNPs')
        plt.hist(shuffle_sad[:,ti,:].flatten(), 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[2], linewidth=1, label='Shuffle')
        ax = plt.gca()
        ax.grid(True, linestyle=':')
        ax.set_xlim(-.15, .15)
        plt.legend()
        plt.savefig('%s/%s_cdf.pdf' % (options.out_dir,targets[ti]))
        plt.close()

        # plot Q-Q
        true_q = mquantiles(true_sad[:,ti], np.linspace(0,1,min(10000,true_sad.shape[0])))
        shuf_q = mquantiles(shuffle_sad_mean[:,ti], np.linspace(0,1,min(10000,true_sad.shape[0])))
        plt.figure()
        plt.scatter(true_q, shuf_q, color=sns_colors[0])
        pmin = 1.05*min(true_q[0], shuf_q[0])
        pmax = 1.05*max(true_q[-1], shuf_q[-1])
        plt.plot([pmin,pmax], [pmin,pmax], color='black', linewidth=1)
        ax = plt.gca()
        ax.set_xlim(pmin,pmax)
        ax.set_ylim(pmin,pmax)
        ax.set_xlabel('True SAD')
        ax.set_ylabel('Shuffled SAD')
        ax.grid(True, linestyle=':')
        plt.savefig('%s/%s_qq.pdf' % (options.out_dir,targets[ti]))
        plt.close()

        # compute Mann-Whitney
        mw_z, mw_p = stats.mannwhitneyu(true_sad[:,ti], shuffle_sad[:,ti,:].flatten())
        cols = (ti, targets[ti], true_sad.shape[0], true_sad[:,ti].mean(), shuffle_sad[:,ti,:].mean(), mw_z, mw_p)
        print('%3d  %20s  %5d  %7.4f  %7.4f  %6.2f  %6.1e' % cols, file=mw_out)

    mw_out.close()


def compute_sad(vcf_file, model_file, out_dir, seq_len, gpu, replot):
    ''' Run basset_sad.py to compute scores. '''

    cuda_str = ''
    if gpu:
        cuda_str = '--cudnn'

    cmd = 'basset_sad.py %s -l %d -o %s %s %s' % (cuda_str, seq_len, out_dir, model_file, vcf_file)
    if not replot:
        subprocess.call(cmd, shell=True)

    sad_table = []
    sad_table_in = open('%s/sad_table.txt' % out_dir)
    sad_table_in.readline()
    last_snpid = None
    for line in sad_table_in:
        a = line.split()
        snpid = a[0]
        sad = float(a[-1])

        if last_snpid == snpid:
            sad_table[-1].append(sad)
        else:
            sad_table.append([sad])

        last_snpid = snpid

    return np.array(sad_table)


def exclude_vcf(vcf_file, excl_bed_file, excl_vcf_file):
    ''' Filter for SNPs outside of the excluded regions
        and remove indels. '''

    # copy header
    excl_vcf_out = open(excl_vcf_file, 'w')
    for line in open(vcf_file):
        if line.startswith('#'):
            print(line, file=excl_vcf_out, end='')
        else:
            break

    # intersect
    p = subprocess.Popen('bedtools intersect -v -a %s -b %s' % (vcf_file, excl_bed_file), stdout=subprocess.PIPE, shell=True)

    for line in p.stdout:
        a = line.split()

        # filter for SNPs only
        if len(a[3]) == len(a[4]) == 1:
            print(line, file=excl_vcf_out, end='')

    excl_vcf_out.close()


def shuffle_snps(vcf_file, shuf_vcf_file, excl_bed_file, genome_file, genome_open):
    ''' Shuffle the given SNPs. '''

    # extract header
    header_lines = []
    for line in open(vcf_file):
        if line.startswith('#'):
            header_lines.append(line)
        else:
            break

    # open shuffled VCF
    shuf_vcf_out = open(shuf_vcf_file, 'w')

    # unset SNPs
    unset_vcf_file = vcf_file
    unset = 1 # anything > 0

    si = 0
    while unset > 0:
        print('Shuffle %d, %d remain' % (si, unset))

        # shuffle w/ BEDtools
        cmd = 'bedtools shuffle -excl %s -i %s -g %s' % (excl_bed_file, unset_vcf_file, genome_file)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

        # update and open next unset VCF
        unset_vcf_file = '%s.%d' % (shuf_vcf_file,si)
        unset_vcf_out = open(unset_vcf_file, 'w')

        # print header
        for line in header_lines:
            print(line, file=unset_vcf_out, end='')

        # zero unset counter
        unset = 0

        # fix alleles before printing
        for line in p.stdout:
            a = line.split()
            chrom = a[0]
            pos = int(a[1])
            snp_nt = a[3]

            # get reference allele
            ref_nt = genome_open.fetch(chrom, pos-1, pos)
            if ref_nt == snp_nt:
                # save to final VCF
                print(line, file=shuf_vcf_out, end='')
            else:
                # write to next unset
                print(line, file=unset_vcf_out, end='')
                unset += 1

        unset_vcf_out.close()

        si += 1

    shuf_vcf_out.close()

    # clean up temp files
    for ci in range(si):
        os.remove('%s.%d' % (shuf_vcf_file,ci))


def shuffle_snps_old(vcf_file, shuf_vcf_file, excl_bed_file, genome_file, genome_open):
    ''' Shuffle the given SNPs. '''

    # open shuffled VCF
    shuf_vcf_out = open(shuf_vcf_file, 'w')

    # shuffle w/ BEDtools
    cmd = 'bedtools shuffle -excl %s -i %s -g %s' % (excl_bed_file, vcf_file, genome_file)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

    # fix alleles before printing
    for line in p.stdout:
        a = line.split()
        chrom = a[0]
        pos = int(a[1])
        snp_nt = a[3]

        # set reference allele
        ref_nt = genome_open.fetch(chrom, pos-1, pos)

        # I accidentally deleted sampling the alt_nt

        # write into column
        a[3] = ref_nt
        a[4] = alt_nt
        print('\t'.join(a), file=shuf_vcf_out)

    shuf_vcf_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
