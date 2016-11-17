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
from scipy.stats import binom
from scipy.stats.mstats import mquantiles
import seaborn as sns

import stats

################################################################################
# basset_sick_loss.py
#
# Shuffle SNPs that overlap DNase sites within their sites and compare the SAD
# distributions.
#
# Todo:
#  -Control for GC% changes introduced by mutation shuffles.
#  -Control for positional changes within the DHS regions.
#  -Properly handle indels.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <vcf_file> <sample_beds_file> <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta', default='%s/assembly/hg19.fa'%os.environ['HG19'], help='Genome FASTA [Default: %default]')
    parser.add_option('-g', dest='gpu', default=False, action='store_true', help='Run on GPU [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad_shuffle', help='Output directory')
    parser.add_option('-r', dest='replot', default=False, action='store_true', help='Re-plot only, without re-computing [Default: %default]')
    parser.add_option('-s', dest='num_shuffles', default=1, type='int', help='Number of SNP shuffles [Default: %default]')
    parser.add_option('-t', dest='sad_table_file', help='Pre-computed SAD scores for the SNPs')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide VCF file, sample BEDs file, and model file')
    else:
        vcf_file = args[0]
        sample_beds_file = args[1]
        model_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # open reference genome
    genome = pysam.Fastafile(options.genome_fasta)

    # open binomial stats file
    binom_out = open('%s/binom.txt' % options.out_dir, 'w')

    # open mann-whitney stats file
    mw_out = open('%s/mannwhitney.txt' % options.out_dir, 'w')

    # plot defaults
    sns.set(font_scale=1.5, style='ticks')

    si = 0
    for line in open(sample_beds_file):
        sample, bed_file = line.split()
        print(sample)

        #########################################
        # compute SAD
        #########################################
        # filter VCF to overlapping SNPs
        print("  intersecting SNPs")
        sample_vcf_file = '%s/%s.vcf' % (options.out_dir,sample)
        if not options.replot:
            filter_vcf(vcf_file, bed_file, sample_vcf_file)

        # compute SAD scores for this sample's SNPs
        print("  computing SAD")
        if options.sad_table_file:
            true_sad = retrieve_sad(sample_vcf_file, options.sad_table_file, si)
        else:
            true_sad = compute_sad(sample_vcf_file, model_file, si, '%s/%s_sad'%(options.out_dir,sample), options.seq_len, options.gpu, options.replot)

        #########################################
        # compute shuffled SAD
        #########################################
        shuffle_sad = np.zeros((true_sad.shape[0],options.num_shuffles))
        for ni in range(options.num_shuffles):
            # shuffle the SNPs within their overlapping DHS
            print("  shuffle %d" % ni)
            sample_vcf_shuf_file = '%s/%s_shuf%d.vcf' % (options.out_dir, sample, ni)
            shuffle_snps(sample_vcf_file, sample_vcf_shuf_file, genome)

            # compute SAD scores for shuffled SNPs
            print("  computing shuffle SAD")
            shuffle_sad[:,ni] = compute_sad(sample_vcf_shuf_file, model_file, si, '%s/%s_shuf%d_sad'%(options.out_dir,sample,ni), options.seq_len, options.gpu, options.replot)

        #########################################
        # simple stats
        #########################################
        # compute shuffle means
        shuffle_sad_mean = shuffle_sad.mean(axis=1)

        # print sample table
        sample_sad_out = open('%s/%s_table.txt' % (options.out_dir,sample), 'w')
        for vi in range(len(true_sad)):
            print('%f\t%f' % (true_sad[vi], shuffle_sad_mean[vi]), file=sample_sad_out)
        sample_sad_out.close()

        # scatter plot
        # plt.figure()
        # plt.scatter(true_sad, shuffle_sad_mean, color='black', alpha=0.7)
        # plt.gca().grid(True, linestyle=':')
        # plt.savefig('%s/%s_scatter.pdf' % (options.out_dir,sample))
        # plt.close()

        # plot CDFs
        sns_colors = sns.color_palette('deep')
        plt.figure()
        plt.hist(true_sad, 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[0], linewidth=1, label='SNPs')
        plt.hist(shuffle_sad.flatten(), 1000, normed=1, histtype='step', cumulative=True, color=sns_colors[2], linewidth=1, label='Shuffle')
        ax = plt.gca()
        ax.grid(True, linestyle=':')
        ax.set_xlim(-.2, .2)
        plt.legend()
        plt.savefig('%s/%s_cdf.pdf' % (options.out_dir,sample))
        plt.close()

        # plot Q-Q
        true_q = mquantiles(true_sad, np.linspace(0,1,min(10000,true_sad.shape[0])))
        shuf_q = mquantiles(shuffle_sad_mean, np.linspace(0,1,min(10000,true_sad.shape[0])))
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
        plt.savefig('%s/%s_qq.pdf' % (options.out_dir,sample))
        plt.close()


        #########################################
        # statistical tests
        #########################################
        # compute matched binomial test
        true_great = sum((true_sad-shuffle_sad_mean) > 0)
        true_lo = np.log2(true_great) - np.log2(len(true_sad)-true_great)
        if true_lo > 0:
            binom_p = 1.0 - binom.cdf(true_great-1, n=len(true_sad), p=0.5)
        else:
            binom_p = binom.cdf(true_great, n=len(true_sad), p=0.5)

        # print significance stats
        cols = (sample, len(true_sad), true_great, true_lo, binom_p)
        print('%-20s  %5d  %5d  %6.2f  %6.1e' % cols, file=binom_out)

        # compute Mann-Whitney
        mw_z, mw_p = stats.mannwhitneyu(true_sad, shuffle_sad.flatten())
        cols = (sample, len(true_sad), true_sad.mean(), shuffle_sad.mean(), mw_z, mw_p)
        print('%-20s  %5d  %6.3f  %6.3f  %6.2f  %6.1e' % cols, file=mw_out)

        # update sample index
        si += 1

    binom_out.close()
    mw_out.close()
    genome.close()


def compute_sad(sample_vcf_file, model_file, si, out_dir, seq_len, gpu, replot):
    ''' Run basset_sad.py to compute scores. '''

    cuda_str = ''
    if gpu:
        cuda_str = '--cudnn'

    cmd = 'basset_sad.py %s -l %d -o %s %s %s' % (cuda_str, seq_len, out_dir, model_file, sample_vcf_file)
    if not replot:
        subprocess.call(cmd, shell=True)

    sad = []
    for line in open('%s/sad_table.txt' % out_dir):
        a = line.split()
        if a[3] == 't%d'%si:
            sad.append(float(a[-1]))

    return np.array(sad)


def filter_vcf(vcf_file, bed_file, sample_vcf_file):
    ''' Filter the VCF file for SNPs that overlap
         the BED file, removing indels. '''

    # open filtered file
    sample_vcf_out = open(sample_vcf_file, 'w')

    # intersect
    p = subprocess.Popen('bedtools intersect -wo -a %s -b %s' % (vcf_file, bed_file), stdout=subprocess.PIPE, shell=True)

    for line in p.stdout:
        a = line.split()
        if len(a[3]) == len(a[4]) == 1:
            print(line, file=sample_vcf_out, end='')

    sample_vcf_out.close()


def retrieve_sad(sample_vcf_file, sad_table_file, si):
    ''' Retrieve SAD scores from a pre-computed table.

        Note that I'm assuming here the table has all
        SAD scores in one row for each SNP so I can
        pull out the score I want as column si+1.
    '''

    snp_indexes = {}
    vi = 0
    for line in open(sample_vcf_file):
        a = line.split()
        snp_indexes[a[2]] = vi
        vi += 1

    sad = np.zeros(len(snp_indexes))
    for line in open(sad_table_file):
        a = line.split()
        print(a)
        if a[0] in snp_indexes:
            sad[snp_indexes[a[0]]] = float(a[si+1])

    return sad


def shuffle_snps(in_vcf_file, out_vcf_file, genome):
    ''' Shuffle the SNPs within their overlapping DHS. '''
    out_vcf_open = open(out_vcf_file, 'w')

    for line in open(in_vcf_file):
        a = line.split()

        # read SNP info
        snp_chrom = a[0]
        snp_pos = int(a[1])
        snp_nt = a[3]

        # determine BED start
        bi = 5
        while a[bi] != snp_chrom:
            bi += 1

        # read BED info
        bed_chrom = a[bi]
        bed_start = int(a[bi+1])
        bed_end = int(a[bi+2])

        # get sequence
        bed_seq = genome.fetch(bed_chrom, bed_start, bed_end)

        # determine matching positions
        bed_nt_matches = [i for i in range(len(bed_seq)) if bed_seq[i] == snp_nt]
        while len(bed_nt_matches) == 0:
            # expand segment by 10 nt
            bed_start = max(0, bed_start-10)
            bed_end += 10
            bed_seq = genome.fetch(bed_chrom, bed_start, bed_end)

        # sample new SNP position
        shuf_pos = bed_start + 1 + random.choice(bed_nt_matches)

        # write into columns
        a[1] = str(shuf_pos)
        print('\t'.join(a), file=out_vcf_open)

    out_vcf_open.close()


def shuffle_snps_old(in_vcf_file, out_vcf_file, genome):
    ''' Shuffle the SNPs within their overlapping DHS. '''
    out_vcf_open = open(out_vcf_file, 'w')
    for line in open(in_vcf_file):
        a = line.split()

        # read SNP info
        snp_chrom = a[0]
        snp_pos = int(a[1])

        # determine BED start
        bi = 5
        while a[bi] != snp_chrom:
            bi += 1

        # read BED info
        bed_chrom = a[bi]
        bed_start = int(a[bi+1])
        bed_end = int(a[bi+2])

        # sample new SNP position
        shuf_pos = random.randint(bed_start, bed_end-1)
        while shuf_pos == snp_pos:
            shuf_pos = random.randint(bed_start, bed_end-1)

        # set reference allele
        ref_nt = genome.fetch(snp_chrom, shuf_pos-1, shuf_pos)

        # sample alternate allele
        alt_nt = random.choice('ACGT')
        while alt_nt == ref_nt:
            alt_nt = random.choice('ACGT')

        # write into columns
        a[1] = str(shuf_pos)
        a[3] = ref_nt
        a[4] = alt_nt
        print('\t'.join(a), file=out_vcf_open)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
