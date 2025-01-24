""" Compute PCA Transform on data stored in CSV format 
"""

import os
import numpy as np


def read_csv_data(data_path):
    """ Read input data stored as csv
        Parameters
        ----------
        data_path : str
            path to the input .csv file.
            expecting data to be float without NaN.

        Returns
        -------
        np.array
            input data stored in numpy array
        
        Raises
        ------
        FileNotFoundError
            wrong data_path
    """
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f'{data_path} file does not exist')


def substract_mean(mat):
    """Compute the mean over all the samples and substract it from the data
    See
    https://stackoverflow.com/questions/22053050/difference-between-numpy-array-shape-r-1-and-r

        Parameters
        ----------
        data : np.array
            dataset stored as a numpy array

        Returns 
        -------
        np.array
            dataset with zero mean
    """


def get_covariance_matrix(data):
    """ Compute the covariance matrix over the dataset
        The covariance matrix is computed as
            1/n_samples * (transpose(_mat) * _mat)
        where _mat is matrix data centered


        Parameters
        ----------
        data : np.array
            dataset stored as a numpy array.

        Returns
        -------
        np.array
            covariance matrix

        Raises
        ------
        AssertionError
            data has not been centered before computing covariance
    """
    assert np.isclose(np.mean(data), 0), "data is not centered"


def eigenvectors(cov_mat):
    """ Compute eigenvectors on input dataset using numpy svd

        Parameters
        ----------
        cov_mat : np.array
            covariance matrix

        Returns
        -------
        2D np.array
            eigen vectors

        1D np.array
            eigen values
            
        Raises
        ------
        AssertionError
            cov_mat is not square matrix (wrong matrix ?)

    """
    assert cov_mat.shape[0] == cov_mat.shape[1], "matrix is not square, not a covariance matrix"


def sort_values(e_vals, e_vecs):
    """ Sort eigen vectors and eigen values in decreasing order
        Select which dimensions to keep using some criterion
        (ex: threshold of how much data explained)
        TODO flip sign if negative ?

        Parameters
        ----------
        e_vals : np.array
            eigen values

        Returns
        -------
        1D np.array
            sorted index

    """


def apply_transform():
    """Apply transformation to input data
    """

def main():
    """ Load input csv, compute the covariance matrix on input data, compute the eigen decomposition on covariance matrix and select principal components.
    """

if __name__ == "__main__":
    main()
