import h5py
import numpy as np


def generate_data(path, batch_size):
    f = h5py.File(path, 'r')

    def train_gen():
        count = 0
        train_in = f['train_in']
        train_out = f['train_out']
        while True:
            count = count + 1
            start = np.random.randint(train_in.shape[0] - batch_size)
            tri = train_in[start:(start + batch_size), :, 0, :].transpose([0, 2, 1])
            tro = train_out[start:(start + batch_size), :]
            yield tri, tro

    def valid_gen():
        valid_in = f['valid_in']
        valid_out = f['valid_out']
        while True:
            for start in range(0, valid_in.shape[0], batch_size):
                vi = valid_in[start:(start + batch_size), :, 0, :].transpose([0, 2, 1])
                vo = valid_out[start:(start + batch_size), :]
                yield vi, vo
            yield None, None

    return f['train_in'].shape[0], train_gen(), valid_gen()
