#!/usr/bin/env python
from optparse import OptionParser
import os, subprocess, tempfile

import h5py

import dna_io

'''
basset_predict.py

Predict a set of test sequences.
'''

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <input_file> <output_file>'
    parser = OptionParser(usage)
    parser.add_option('-b', '--batch', dest='batch', default=128, type='int', help='Batch size [Default: %default]')
    parser.add_option('--cuda', dest='cuda', default=False, action='store_true', help='Run on GPGPU [Default: %default]')
    parser.add_option('--cudnn', dest='cudnn', default=False, action='store_true', help='Run on GPGPU w/cuDNN [Default: %default]')
    parser.add_option('-n', '--norm', dest='norm', default=False, action='store_true', help='Normalize all targets to a level plane [Default: %default]')
    parser.add_option('-r', '--rc', dest='rc', default=False, action='store_true', help='Average forward and reverse complement [Default: %default]')
    parser.add_option('-s', '--scores', dest='scores', default=False, action='store_true', help='Print pre-sigmoid scores instead of probability predictions [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide Basset model file, input sequences (as a FASTA file or test data in an HDF file, and output table file')
    else:
        model_file = args[0]
        input_file = args[1]
        out_file = args[2]

    #################################################################
    # parse input file
    #################################################################
    try:
        # input_file is FASTA

        # load sequences
        seqs_1hot = dna_io.load_sequences(input_file, permute=False)

        # reshape sequences for torch
        seqs_1hot = seqs_1hot.reshape((seqs_1hot.shape[0],4,1,seqs_1hot.shape[1]/4))

        # write as test data to a HDF5 file
        model_input_fd, model_input_hdf5 = tempfile.mkstemp()
        h5f = h5py.File(model_input_hdf5, 'w')
        h5f.create_dataset('test_in', data=seqs_1hot)
        h5f.close()
        temp_hdf5 = True

    except (IOError, IndexError):
        # input_file is HDF5
        model_input_hdf5 = input_file
        temp_hdf5 = False


    #################################################################
    # Torch predict modifications
    #################################################################
    opts_str = '-batch %d' % options.batch
    if options.cudnn:
        opts_str += ' -cudnn'
    elif options.cuda:
        opts_str += ' -cuda'
    if options.norm:
        opts_str += ' -norm'
    if options.rc:
        opts_str += ' -rc'
    if options.scores:
        opts_str += ' -scores'

    torch_cmd = 'basset_predict.lua %s %s %s %s' % (opts_str, model_file, model_input_hdf5, out_file)
    print(torch_cmd)
    subprocess.call(torch_cmd, shell=True)

    if temp_hdf5:
        os.remove(model_input_hdf5)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
