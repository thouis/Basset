#!/usr/bin/env python

from optparse import OptionParser
import os
import subprocess

################################################################################
# basset_sat_sad.py
#
# Make saturated mutagenesis plots for a subset of SNPs, determined by some
# simple rules from a SAD table.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file> <vcf_file> <sad_table>'
    parser = OptionParser(usage)
    parser.add_option('-n', dest='top_n', default=1, type='int', help='Plot max top_n targets for SNPs w/ SAD > sad_t [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sat_sad')
    parser.add_option('-s', dest='sad_t', default=0.1, type='float', help='Plot SNP/targets with SAD > sad_t [Default: %default]')
    parser.add_option('-t', dest='target_labels')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide Basset model file, input SNPs in VCF format, and SAD table')
    else:
        model_file = args[0]
        vcf_file = args[1]
        sad_file = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    # hash SAD by SNP
    sad_in = open(sad_file)
    sad_in.readline()
    snp_sads = {}
    for line in sad_in:
        sad = SAD(line)
        snp_sads.setdefault(sad.snp,[]).append(sad)
    sad_in.close()

    # sort SNPs by max SAD
    snp_max = []
    for snp in snp_sads:
        sad_max = max([abs(s.sad) for s in snp_sads[snp]])
        snp_max.append((sad_max,snp))

    snp_max.sort(reverse=True)

    # map target labels to indexes
    target_indexes = {}
    ti = 0
    for line in open(options.target_labels):
        a = line.split()
        target_indexes[a[0]] = ti
        ti += 1

    # plot top
    for sad_max, snp in snp_max:
        if sad_max > options.sad_t:
            # decide which to plot
            abs_sads = sorted([(abs(s.sad),s) for s in snp_sads[snp]], reverse=True)
            plot_targets = []
            for ni in range(options.top_n):
                sad_target = abs_sads[ni][1].target
                ti = target_indexes[sad_target]
                plot_targets.append(str(ti))

            # get VCF line
            snp_vcf_file = '%s/%s.vcf' % (options.out_dir, snp)
            snp_vcf_out = open(snp_vcf_file, 'w')
            for line in open(vcf_file):
                a = line.split()
                if a[2] == snp:
                    print(line, file=snp_vcf_out, end='')
            snp_vcf_out.close()

            # plot
            plot_out_dir = '%s/%s_sat' % (options.out_dir, snp)
            cmd = 'basset_sat_vcf.py -l 1000 -o %s -t %s %s %s' % (plot_out_dir, ','.join(plot_targets), model_file, snp_vcf_file)
            subprocess.call(cmd, shell=True)


class SAD:
    def __init__(self, sad_line):
        a = sad_line.split()
        self.snp = a[0]
        self.target = a[3]
        self.sad = float(a[6])

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
