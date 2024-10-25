""" Compute PCA Transform on data stored in CSV format 
"""

import ipdb
import numpy as np


def generate_data(dim1, dim2):
    """Generate Random Data with size dim1 x dim2.
    Data are store in a numpy array
    """
    mat = np.random.rand(dim1, dim2)
    return mat


def read_csv_data(data_path):
    """Read spike data and store as numpy array"""


def substract_mean(mat):
    """Compute the mean over all the samples and substract it from the data
    See
    https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r
    """
    # Allocate array
    mu = np.zeros((mat.shape[1],))
    for row in mat:
        mu += row
    return mat - 1 / mat.shape[0] * mu


def get_covariance_matrix(centered_data):
    """Compute the covariance matrix over the dataset
    The covariance matrix is computed as
        1/n_samples * (transpose(mat) * mat)
    assuming the mean of mat is 0
    """
    dim1, _ = centered_data.shape
    return 1 / dim1 * np.matmul(np.transpose(centered_data), centered_data)


def eigenvectors(cov_mat):
    """Compute eigenvectors on input dataset using numpy svd"""
    # U, S, V = np.linalg.svd(cov_mat)
    e_vals, e_vecs = np.linalg.eig(cov_mat)

    return e_vals, e_vecs


def sort_values(e_vals, e_vecs):
    """Select which dimensions to keep using some criterion
    (ex: threshold of how much data explained)
    """
    index = np.flip(np.argsort(e_vals))

    # check if some values are negative ?
    for eigenval in e_vals:
        if eigenval < 0:
            print(eigenval)

    return index


def apply_transform():
    """Apply transformation to input data"""

def write_output():
    """Write transformed data"""

def whitening():
    """Whiten PCA output transform"""


def main():
    """Load input matrix or generate a random one,
    compute its covariance matrix, then compute its eigenvectors using numpy svd
    """
    from sklearn.decomposition import PCA

    mat = generate_data(100, 1000)

    centered_data = substract_mean(mat)

    cov_mat = get_covariance_matrix(centered_data)
    e_val, e_vec = eigenvectors(cov_mat)
    ipdb.set_trace()


if __name__ == "__main__":
    main()
