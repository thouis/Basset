#!/usr/bin/env python
from optparse import OptionParser
from collections import OrderedDict
import subprocess
import sys

import numpy as np
import pyBigWig

################################################################################
# preprocess_chromatin.py
#
# Annotate the sites in a BED file with a set of Wig/BigWig tracks.
################################################################################


################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <sites_bed> <sample_wigs_file> <out_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', dest='bin_size', default=None, type='int', help='Bin size to take the mean track value [Default: %default]')
    parser.add_option('--clip_min', dest='clip_min', default=None, type='float', help='Clip the distribution minimums at this proportion. [Default: %default; Suggested: 0.05]')
    parser.add_option('--clip_max', dest='clip_max', default=None, type='float', help='Clip the distribution maximums at this proportion [Default: %default; Suggested: .001]')
    parser.add_option('-f', dest='function', default='mean', help='Function to compute in each bin [Default: %default]')
    parser.add_option('-l', dest='log2', default=False, action='store_true', help='Take log2 [Default: %default')
    parser.add_option('-n', dest='normalize', default=False, action='store_true', help='Normalize [Default: %default]')
    parser.add_option('-p', dest='pseudocount', default=0, type='float', help='Pseudocount added before transformations [Default: %default]')
    parser.add_option('-s', dest='span', default=200, type='int', help='Span of sequence to consider around each site Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide sites BED and file with Wig/BigWig labels and paths')
    else:
        sites_bed_file = args[0]
        sample_wigs_file = args[1]
        out_file = args[2]

    if options.bin_size is None:
        options.bin_size = options.span

    if options.span % options.bin_size != 0:
        parser.error('Bin size must evenly divide span')

    #################################################################
    # setup
    #################################################################
    # determine span on each side
    span_left = int(round(options.span/2.0))
    span_right = options.span/2

    # read in sites
    sites = []
    for line in open(sites_bed_file):
        a = line.split()
        start = int(a[1])
        end = int(a[2])
        mid = (start + end)/2

        sites.append(Site(a[0], mid-span_left, mid+span_right))

    # count bins
    num_bins = options.span / options.bin_size

    # get wig files and labels
    target_wigs = OrderedDict()
    for line in open(sample_wigs_file):
        a = line.split()
        target_wigs[a[0]] = a[1]

    # label features
    feature_labels = []
    for target in list(target_wigs.keys()):
        if num_bins == 1:
            feature_labels.append(target)
        else:
            bin_mid = -span_left + options.bin_size/2
            for bi in range(num_bins):
                feature_labels.append('%s_pos%d' % (target,bin_mid))
                bin_mid += options.bin_size

    # initialize features array
    site_features = np.zeros((len(sites),len(feature_labels)), dtype='float16')

    #################################################################
    # process wigs
    #################################################################
    wi = 0
    for sample in target_wigs:
        wig_file = target_wigs[sample]
        print(wig_file)

        # open wig
        wig_in = pyBigWig.open(wig_file)

        for si in range(len(sites)):
            s = sites[si]

            # pull stats from wig
            bin_stats = wig_in.stats(s.chrom, s.start, s.end, type=options.function, nBins=num_bins)

            # copy into primary data structure
            for bi in range(num_bins):
                site_features[si,wi+bi] = bin_stats[bi]

        wi += num_bins

    #################################################################
    # normalize
    #################################################################
    # replace nan with zero
    site_features = np.nan_to_num(site_features)

    # add pseudocount
    if options.pseudocount:
        site_features += options.pseudocount

    # acknowledge the max of float16
    site_features = site_features.clip(0, 65504.0)

    if options.log2:
        site_features = np.log2(site_features)

    # normalize
    if options.normalize:
        site_features = site_features - np.nanmean(site_features, axis=0)
        site_features = site_features / np.nanstd(site_features, axis=0)

    # clip minimum
    if options.clip_min is not None:
        wmins = np.percentile(site_features, 100*options.clip_min, axis=0)
        site_features = site_features.clip(min=wmins)

    # clip maximum
    if options.clip_max is not None:
        wmaxs = np.percentile(site_features, 100*(1-options.clip_max), axis=0)
        site_features = site_features.clip(max=wmaxs)

    #################################################################
    # output
    #################################################################
    sv_out = open(out_file, 'w')

    cols = [''] + feature_labels
    print('\t'.join(cols), file=sv_out)

    si = 0
    for line in open(sites_bed_file):
        a = line.split()
        site_id = '%s:%s-%s(%s)' % (a[0], a[1], a[2], a[5])
        cols = [site_id] + [str(v) for v in site_features[si]]
        print('\t'.join(cols), file=sv_out)
        si += 1

    sv_out.close()


class Site:
    def __init__(self, chrom, start, end):
        self.chrom = chrom
        self.start = start
        self.end = end

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
