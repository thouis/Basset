#!/usr/bin/env python
from optparse import OptionParser
import os
import random
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.externals import joblib
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

################################################################################
# basset_postmodel.py
#
# Train the final layer of the model with additional data.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <repr_hdf5> <data_hdf5> <target_index>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='add_only', default=False, action='store_true', help='Use additional features only; no sequence features')
    parser.add_option('-b', dest='balance', default=False, action='store_true', help='Downsample the negative set to balance [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='postmodel', help='Output directory [Default: %default]')
    parser.add_option('-r', dest='regression', default=False, action='store_true', help='Regression mode [Default: %default]')
    parser.add_option('-s', dest='seq_only', default=False, action='store_true', help='Use sequence features only; no additional features [Default: %default]')
    parser.add_option('--sample', dest='sample', default=None, type='int', help='Sample from the training set [Default: %default]')
    parser.add_option('-t', dest='target_hdf5', default=None, help='Extract targets from this HDF5 rather than data_hdf5 argument')
    parser.add_option('-x', dest='regex_add', default=None, help='Filter additional features using a comma-separated list of regular expressions')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide full data HDF5, representation HDF5, and target index or filename')
    else:
        repr_hdf5_file = args[0]
        data_hdf5_file = args[1]
        target_i = args[2]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(1)

    #######################################################
    # preprocessing
    #######################################################

    # load training targets
    data_hdf5_in = h5py.File(data_hdf5_file, 'r')
    if options.target_hdf5:
        target_hdf5_in = h5py.File(options.target_hdf5, 'r')
    else:
        target_hdf5_in = data_hdf5_in
    train_y = np.array(target_hdf5_in['train_out'])[:,target_i]
    test_y = np.array(target_hdf5_in['test_out'])[:,target_i]

    # load training representations
    if not options.add_only:
        repr_hdf5_in = h5py.File(repr_hdf5_file, 'r')
        train_x = np.array(repr_hdf5_in['train_repr'])
        test_x = np.array(repr_hdf5_in['test_repr'])
        repr_hdf5_in.close()

    if options.seq_only:
        add_labels = []

    else:
        # load additional features
        train_a = np.array(data_hdf5_in['train_add'])
        test_a = np.array(data_hdf5_in['test_add'])
        add_labels = np.array(data_hdf5_in['add_labels'])

        if options.regex_add:
            fi = filter_regex(options.regex_add, add_labels)
            train_a, test_a, add_labels = train_a[:,fi], test_a[:,fi], add_labels[fi]

        # append additional features
        if options.add_only:
            add_i = 0
            train_x, test_x = train_a, test_a
        else:
            add_i = train_x.shape[1]
            train_x = np.concatenate((train_x,train_a), axis=1)
            test_x = np.concatenate((test_x,test_a), axis=1)

    data_hdf5_in.close()
    if options.target_hdf5:
        target_hdf5_in.close()

    # balance
    if options.balance:
        train_x, train_y = balance(train_x, train_y)

    # sample
    if options.sample is not None and options.sample < train_x.shape[0]:
        sample_indexes = random.sample(list(range(train_x.shape[0])), options.sample)
        train_x = train_x[sample_indexes]
        train_y = train_y[sample_indexes]


    #######################################################
    # model
    #######################################################
    if options.regression:
        # fit
        model = BayesianRidge(fit_intercept=True)
        model.fit(train_x, train_y)

        # accuracy
        acc_out = open('%s/r2.txt' % options.out_dir, 'w')
        print(model.score(test_x, test_y), file=acc_out)
        acc_out.close()

        test_preds = model.predict(test_x)

        # plot a sample of predictions versus actual
        plt.figure()
        sns.jointplot(test_preds[:5000], test_y[:5000], joint_kws={'alpha':0.3})
        plt.savefig('%s/scatter.pdf' % options.out_dir)
        plt.close()

        # plot the distribution of residuals
        plt.figure()
        sns.distplot(test_y-test_preds)
        plt.savefig('%s/residuals.pdf' % options.out_dir)
        plt.close()

    else:
        # fit
        model = LogisticRegression(penalty='l2', C=1000)
        model.fit(train_x, train_y)

        # accuracy
        test_preds = model.predict_proba(test_x)[:,1].flatten()
        acc_out = open('%s/auc.txt' % options.out_dir, 'w')
        print(roc_auc_score(test_y, test_preds), file=acc_out)
        acc_out.close()

        # compute and print ROC curve
        fpr, tpr, thresholds = roc_curve(test_y, test_preds)

        roc_out = open('%s/roc.txt' % options.out_dir, 'w')
        for i in range(len(fpr)):
            print('%f\t%f\t%f' % (fpr[i], tpr[i], thresholds[i]), file=roc_out)
        roc_out.close()

        # compute and print precision-recall curve
        precision, recall, thresholds = precision_recall_curve(test_y, test_preds)

        prc_out = open('%s/prc.txt' % options.out_dir, 'w')
        for i in range(len(precision)):
            print('%f\t%f' % (precision[i], recall[i]), file=prc_out)
        prc_out.close()

    # save model
    joblib.dump(model, '%s/model.pkl' % options.out_dir)

    #######################################################
    # analyze
    #######################################################
    # print coefficients table
    coef_out = open('%s/add_coefs.txt' % options.out_dir, 'w')
    for ai in range(len(add_labels)):
        if options.regression:
            coefi = model.coef_[add_i+ai]
        else:
            coefi = model.coef_[0,add_i+ai]
        print(add_labels[ai], coefi, file=coef_out)
    coef_out.close()


def balance(x,y):
    ''' Down sample the negative set to balance. '''

    positives = np.array([i for i in range(y.shape[0]) if y[i] == 1])
    negatives = np.array([i for i in range(y.shape[0]) if y[i] == 0])

    print('%d positives to %d negatives' % (len(positives), len(negatives)))

    if len(negatives) < len(positives):
        xb = x
        yb = y

    else:
        negatives_balanced = random.sample(negatives, len(positives))

        indexes_balanced = sorted(np.concatenate((positives, negatives_balanced)))
        xb = x[indexes_balanced,:]
        yb = y[indexes_balanced]

    return xb, yb


def filter_regex(regex, labels):
    ''' Filter the additional attributes for those
            with labels that fit the regex. '''
    regex_strs = regex.split(',')
    regex_res = []
    for regex_str in regex_strs:
        regex_res.append(re.compile(regex_str))

    filter_indexes = []
    for i in range(len(labels)):
        # compare to regex's
        filter_match = False
        for regex_re in regex_res:
            regex_m = regex_re.search(labels[i])
            if regex_m:
                filter_match = True

        if filter_match:
            filter_indexes.append(i)

    return filter_indexes



################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
