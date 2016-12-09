import sys

import numpy as np
import datagen
from sklearn.metrics import roc_auc_score, average_precision_score

from keras.layers import Convolution1D, MaxPooling1D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
# from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import Callback
from keras.objectives import binary_crossentropy
from keras.regularizers import l1, activity_l1
import keras.backend as K
import theano

from Eve import Eve
from prg import prg


def maybe_print(tensor, msg, do_print=True):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor


def binary_crossentropy_sum(y_true, y_pred):
    return K.mean(K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1))


def F_loss(y_true, y_pred):
    TP = K.sum(y_true * y_pred)
    FN = K.sum(y_true * (1 - y_pred))
    FP = K.sum((1 - y_true) * y_pred)
    return 1 - TP / (TP + 0.5 * FN + 0.5 * FP)


def hinge_loss(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    diffs = y_pred[:, None] - y_pred[None, :]
    pos_minus_neg = y_true[:, None] * (1 - y_true[None, :])
    unweighted_loss = K.clip(0.5 - diffs, 0, 2)
    L = K.sum(unweighted_loss * pos_minus_neg) / K.sum(pos_minus_neg)
    return L


def AUC_loss(y_true, y_pred):
    y_false = 1 - y_true
    mean_true = K.sum(y_true * y_pred) / K.sum(y_true)
    mean_false = K.sum(y_false * y_pred) / K.sum(y_false)
    return 1 - mean_true * (1 - mean_false)

def AUC_weighted_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_false = 1 - y_true
    w_true = y_true * (1 - y_pred + 0.1)
    mean_true = K.sum(w_true * y_pred) / K.sum(w_true)
    w_false = y_false * (y_pred + 0.1)
    mean_false = K.sum(w_false * y_pred) / K.sum(w_false)
    return 1 - mean_true * (1 - mean_false)


def basset_network(input_shape, outputs, act_w=0.1):
    # From original basset repo, best model:
    # conv_filters    300
    # conv_filters    200
    # conv_filters    200
    # conv_filter_sizes       19
    # conv_filter_sizes       11
    # conv_filter_sizes       7
    # pool_width              3
    # pool_width              4
    # pool_width              4
    # hidden_units            1000
    # hidden_units            1000
    # hidden_dropouts         0.3
    # hidden_dropouts         0.3
    # learning_rate           0.002
    # weight_norm             7
    # momentum                0.98

    model = Sequential()

    conv_depths = [300, 200, 200]
    conv_widths = [19, 11, 7]
    pool_widths = [3, 4, 4]

    for cd, cw, pw in zip(conv_depths, conv_widths, pool_widths):
        if input_shape is not None:
            model.add(Convolution1D(cd, cw, input_shape=input_shape))
        else:
            model.add(Convolution1D(cd, cw))
        model.add(BatchNormalization(axis=2))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pw))

        input_shape = None  # after first layer, input shape is inferred

    hidden_depths = [1000, 1000]
    dropouts = [0.3, 0.3]

    model.add(Flatten())
    model.add(Dropout(0.3))
    for hd, dr in zip(hidden_depths, dropouts):
        model.add(Dense(hd))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Dropout(dr))

    model.add(Dense(outputs, activation='sigmoid'))
    return model


class CB(Callback):
    def __init__(self, m, valid_gen, start_epoch=0):
        self.m = m
        self.valid_gen = valid_gen
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs):
        ground_truth = []
        predictions = []
        while True:
            vi, vo = next(self.valid_gen)
            if vi is None:
                break
            vpred = self.m.predict(vi)
            ground_truth.append(vo)
            predictions.append(vpred)

        ground_truth = np.concatenate(ground_truth, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        TP = ((predictions > 0.5) * ground_truth).sum()
        FP = ((predictions > 0.5) * (1 - ground_truth)).sum()
        FN = ((predictions <= 0.5) * ground_truth).sum()
        precision = float(TP) / (TP + FP)
        recall = float(TP) / (TP + FN)
        print("  validation")
        print("     precision  {:f}  recall  {:f}  ({}, {}, {})".
              format(precision, recall, TP, FP, FN))

        AUC = roc_auc_score(ground_truth, predictions, average='macro')
        assert ground_truth.shape[1] == 164
        assert len(ground_truth.shape) == 2
        sum_prg = 0.0
        for idx in range(ground_truth.shape[1]):
            prg_curve = prg.create_prg_curve(ground_truth[:, idx], predictions[:, idx])
            sum_prg += prg.calc_auprg(prg_curve)
        PRC = sum_prg / ground_truth.shape[1]
        print("     AUC {}   PRC {}".format(AUC, PRC))
        print("")
        self.m.save_weights("weights_{}.h5".format(epoch + self.start_epoch))


if __name__ == '__main__':
    batch_size = 64
    epoch_size, train_gen, valid_gen = datagen.generate_data(sys.argv[1], batch_size)
    i, o = next(train_gen)

    model = basset_network(i.shape[1:], o.shape[1], act_w=(0.1 / batch_size))

    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)
    open("model.json", "w").write(model.to_json())

    print("compiling")
    # model.compile(loss=weighted_mse, optimizer=SGD(lr=1e-3, momentum=0.95, clipvalue=0.5))
    opt = Eve(lr=1E-4, decay=1E-4, beta_1=0.9, beta_2=0.999, beta_3=0.999, small_k=0.1, big_K=10, epsilon=1e-08)
    # opt = SGD(lr=0.0001, momentum=0.95)
    # opt = RMSprop(lr=0.02)
    model.compile(loss=AUC_weighted_loss, optimizer=opt)
#    model.load_weights('weights_43.h5')

    print("fitting")
    model.fit_generator(train_gen, (epoch_size // batch_size) * batch_size, 100, verbose=1, callbacks=[CB(model, valid_gen, start_epoch=0)])
