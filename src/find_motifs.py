import sys
import h5py
from keras.models import model_from_json
import keras.backend as K
import numpy as np

if __name__ == '__main__':
    model_file = sys.argv[1]
    model_weights = sys.argv[2]
    sequences_hdf5 = sys.argv[3]
    output_file = sys.argv[4]

    model = model_from_json(open(model_file, "r").read())
    model.load_weights(model_weights)

    layer = model.layers[0]
    print(layer)
    weights = K.get_value(layer.W)
    weights = weights[:, 0, ...].transpose(2, 1, 0)

    outfile = h5py.File(output_file, 'w')
    outfile.create_dataset('weights', data=weights)

    layer = model.layers[1]
    print(layer)
    reprfun = K.function([model.input],
                         [layer.output],
                         givens={K.learning_phase(): np.uint8(0)})

    seq = h5py.File(sequences_hdf5, 'r')['test_in']
    out = None
    print("predicting {} sequences".format(seq.shape[0]))
    for idx in range(seq.shape[0]):
        in_data = seq[idx, :, 0, :].transpose([1, 0])
        repr = reprfun([in_data[np.newaxis, ...]])[0][0, ...]
        repr = repr.transpose((1, 0))

        if out is None:
            out = outfile.create_dataset("outs",
                                         (seq.shape[0], ) + repr.shape,
                                         dtype=np.float32)
        out[idx] = repr
