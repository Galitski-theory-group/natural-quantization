import json
import numpy as np
from natural_quantization.preprocess import read_weights


def test_read():

    data = read_weights("/Users/dlakhdar/physics/repos/natural-quantization/data/quantum_nn_rotation_angles/mnist_a0.5_lr-2_shots20_width16.json")
    data = list( np.array(arr) for arr in data)
    assert type(data) == list

    for arr in data:
        assert type(data) == np.ndarray

    assert data[0].shape == (16,784)
    assert data[1].shape == (16,16)
    assert data[2].shape == (16,16)
    data[3].shape
