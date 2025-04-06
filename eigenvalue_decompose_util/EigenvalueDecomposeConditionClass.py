"""
Eigenvalue Decomposition Configuration Module

This module defines the configuration class for eigenvalue decomposition operations.
It provides settings for controlling the decomposition process, including noise simulation,
matrix properties, and optimization parameters.
"""

import os
from dataclasses import dataclass

import joblib  # type: ignore
import numpy as np


@dataclass(frozen=True)
class EigenValueDecomposeCondition:
    """
    Configuration class for eigenvalue decomposition.

    This class stores parameters and settings for eigenvalue decomposition operations,
    including methods for orthonormalization, noise simulation, and matrix properties.

    Attributes:
        seed: Random seed for reproducibility
        method: Method for finding orthonormal vectors in the subspace
        noisy: Whether to include noise when calculating inner products
        n_shots: Number of shots (when including noise)
        set_diag: Whether to fix diagonal elements of the kernel matrix to 1
        symmetric: Whether to make the kernel matrix symmetric (copy upper triangular part to lower triangular part)
        denoise: Whether to apply correction to the kernel
        lambda_reg: Strength of the regularization term in the optimization problem (when applying correction)
    """
    seed: int  # TODO: Unused
    method: str  # Method for finding orthonormal vectors in the subspace
    noisy: bool  # Whether to include noise when calculating inner products
    n_shots: int | None  # Number of shots (when including noise)
    set_diag: bool  # Whether to fix diagonal elements of the kernel matrix to 1
    symmetric: bool  # Whether to make the kernel matrix symmetric (copy upper triangular part to lower triangular part)
    denoise: bool  # Whether to apply correction to the kernel
    lambda_reg: float | int  # Strength of the regularization term in the optimization problem (when applying correction)

    def get_save_path(self, svm_file_name: str) -> str:
        """
        Generate the full path for saving eigenvalue decomposition data.

        This method creates a directory structure based on the SVM filename
        and the eigenvalue decomposition parameters.

        Args:
            svm_file_name: Path to the SVM model file

        Returns:
            str: Full path for saving eigenvalue decomposition data
        """
        # Directory
        first_folder_name = self.get_top_folder_name()
        second_folder_name = os.path.basename(svm_file_name)[
            :-3
        ]  # Remove the trailing .jb
        folder_name = os.path.join(first_folder_name, second_folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

        # File
        file_name = self.get_file_name()
        path = os.path.join(folder_name, file_name)

        return path

    def get_top_folder_name(self) -> str:
        """
        Get the top-level folder name for eigenvalue decomposition data storage.

        Returns:
            str: The name of the top-level folder
        """
        return "svm_eigenvalue_decompose"

    def get_file_name(self) -> str:
        """
        Generate a filename for eigenvalue decomposition data based on configuration parameters.

        The filename includes all relevant parameters to uniquely identify the configuration.

        Returns:
            str: Formatted filename for eigenvalue decomposition data
        """
        if self.lambda_reg == 0.0:
            lambda_reg = 0
        else:
            lambda_reg = self.lambda_reg
        return f"seed={self.seed}-method={self.method}-noisy={self.noisy}-n_shots={self.n_shots}-set_diag={self.set_diag}-symmetric={self.symmetric}-denoise={self.denoise}-lambda_reg={lambda_reg}.jb"

    def load_saved_data(self, svm_file_name: str):
        """
        Load eigenvalue decomposition data from a saved file.

        Args:
            svm_file_name: Path to the SVM model file used to generate the save path

        Returns:
            The loaded eigenvalue decomposition data
        """
        return joblib.load(self.get_save_path(svm_file_name=svm_file_name))

    @staticmethod
    def load_lambda_k(eigen_data, k_index):
        """
        Load a specific eigenvalue from eigenvalue decomposition data.

        Args:
            eigen_data: Eigenvalue decomposition data
            k_index: Index of the eigenvalue to load

        Returns:
            The k_index-th eigenvalue sorted in descending order of magnitude
        """
        lambda_k = eigen_data["eigenvalues"][
            k_index
        ]  # Get the k_index-th eigenvalue sorted in descending order of magnitude
        return lambda_k

    @staticmethod
    def load_eigenvector_k(eigen_data, k_index):
        """
        Load a specific eigenvector from eigenvalue decomposition data.

        Args:
            eigen_data: Eigenvalue decomposition data
            k_index: Index of the eigenvector to load

        Returns:
            The k_index-th eigenvector
        """
        eigen_vector_k = eigen_data["eigenvectors"][k_index]
        return eigen_vector_k

    def load_lambda_k_list_list(self, svm_file_name, K) -> list[list[float]]:
        """
        Load lists of eigenvalues for each data index.

        This method returns a list of lists containing K eigenvalues for each index.
        The format is [[\lambda_1,...\lambda_K],[\lambda_1,...\lambda_K],[\lambda_1,...\lambda_K],...]

        Args:
            svm_file_name: Path to the SVM model file used to generate the save path
            K: Number of eigenvalues to load for each data index

        Returns:
            List of lists containing K eigenvalues for each data index
        """
        # Get eigenvalues corresponding to K eigenvectors for each data_index
        eigen_datas = self.load_saved_data(svm_file_name=svm_file_name)

        lambda_k_list_list = [
            [
                EigenValueDecomposeCondition.load_lambda_k(
                    eigen_data=eigen_data, k_index=k_index
                )
                for k_index in range(K)
            ]
            for eigen_data in eigen_datas
        ]
        return lambda_k_list_list

    def load_eigenvector_k_list_list(self, svm_file_name, K) -> list[list[np.ndarray]]:
        """
        Load lists of eigenvectors for each data index.

        This method returns a list of lists containing K eigenvectors for each index.

        Args:
            svm_file_name: Path to the SVM model file used to generate the save path
            K: Number of eigenvectors to load for each data index

        Returns:
            List of lists containing K eigenvectors for each data index
        """
        eigen_datas = self.load_saved_data(svm_file_name=svm_file_name)

        eigenvector_k_list_list = [
            [
                EigenValueDecomposeCondition.load_eigenvector_k(
                    eigen_data=eigen_data, k_index=k_index
                )
                for k_index in range(K)
            ]
            for eigen_data in eigen_datas
        ]
        return eigenvector_k_list_list
