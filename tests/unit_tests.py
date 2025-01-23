""" Unit tests for PCA Module"""

import pytest
import numpy as np

from PCA.PCA import *
from sklearn.datasets import load_iris, make_low_rank_matrix

# global_seed ?

# Define different input datasets
@pytest.fixture(scope="module")
def random_data():
    """Use random Matrix"""
    # rng = np.random.RandomState(0)
    np.random.seed(0) # use a seed for reproducibility
    mat = np.random.rand(100, 1000)
    # rng(100, 1000)  # random size ?

    return mat


@pytest.fixture(scope="module")
def iris():
    """Use iris dataset - 
       Classic dataset used to demo PCA or classification problems.
    """
    iris_dataset = load_iris()
    return iris_dataset.data


@pytest.mark.parametrize("data_fixture", ["random_data", "iris"])
def test_data_centered(data_fixture, request):
    """Check that remove_mean returns a matrix of mean 0
       Fixture is used to run same test on several input datasets.
    """
    mat = request.getfixturevalue(data_fixture)

    # data_fixture
    centered_mat = substract_mean(mat)
    assert np.any(
        np.isclose(
            np.mean(substract_mean(centered_mat), axis=0),
            np.zeros((centered_mat.shape[1],)),
        )
    )


def check_variance_dim():
    """Check that the 1st dim is indeed the dimension with the most variance"""


def check_covariance_identity():
    """Covariance Matrix of identity_matrix is 1/identity_matrix.size * identity_matrix"""
    identity_mat = np.eye(100)
    cov_idmat = get_covariance_matrix(identity_mat)
    assert cov_idmat == 1 / identity_mat.shape[0] * identity_mat


# @pytest.mark.parametrize("data_fixture", ["random_data", "iris"])
def test_covariance_correlated():
    """Covariance matrix values of rank 1 matrix are norm of column * factor"""

    # generate random vector
    np.random.seed(0)
    rand_vec = np.random.rand(100)

    # generate rank 1 matrix by stacking this random vector, multiplied by some coefficients
    rank1_matrix = np.transpose(np.stack([(u + 1) * rand_vec for u in range(10)]))
    cov_mat = get_covariance_matrix(rank1_matrix)
    for i in range(10):
        for j in range(i, 10):
            assert np.isclose(
                cov_mat[i, j], np.linalg.norm(rand_vec) ** 2 * (i + 1) * (j + 1) / 100
            )


@pytest.mark.parametrize("data_fixture", ["random_data", "iris"])
def test_covariance_symmetric(data_fixture, request):
    """ Covariance Matrix should alway be symmetric"""
    mat = request.getfixturevalue(data_fixture)
    cov_mat = get_covariance_matrix(mat)
    assert np.all(cov_mat == np.transpose(cov_mat))


@pytest.mark.parametrize("data_fixture", ["random_data", "iris"])
def test_covariance_positive_semidefinite(data_fixture, request):
    """ Covariance Matrix should be positive semidefinite - Test with sylvester's criterion"""
    # rng = np.random.RandomState(0)
    mat = request.getfixturevalue(data_fixture)
    # mat = rng(80, 160)  # random size ?
    cov_mat = get_covariance_matrix(mat)
    for i in range(cov_mat.shape[0]):
        assert np.isclose(np.linalg.det(cov_mat[:i, :i]), 0) or (
            np.linalg.det(cov_mat[:i, :i]) >= 0
        )


def test_covariance_no_correlation():
    pass
