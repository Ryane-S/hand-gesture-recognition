import pytest
import numpy as np

from model.model import prepare_data

@pytest.fixture
def mockup_dataset():
    x_train = np.array([np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for i in range(36)])
    x_test = np.array([np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8) for i in range(8)])
    y_train = np.array([np.random.randint(0, 36, (36, 1), dtype=np.uint8) for i in range(36)])
    y_test = np.array([np.random.randint(0, 36, (36, 1), dtype=np.uint8) for i in range(8)])
    return x_train, y_train, x_test, y_test

def test_model(mockup_dataset):
    x_train, y_train, x_test, y_test = mockup_dataset
    x_train, y_train, x_test, y_test, input_shape, num_classes = prepare_data(x_train, y_train, x_test, y_test)
    assert np.all((x_train>=0) & (x_train <=1))
    assert np.all((x_test>=0) & (x_test <=1))
    assert np.all((y_train==0) | (y_train==1))
    assert np.all((y_test==0) | (y_test==1))
    assert input_shape == (224,224,3)
    assert num_classes == 36
