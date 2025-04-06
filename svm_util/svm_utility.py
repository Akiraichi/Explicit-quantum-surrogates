from typing import Any

import numpy as np
from jaxtyping import Complex128

from misc.DatasetClass import Dataset
from misc.kernel import fidelity_kernel, get_kernel_matrix
from svm_util.SVMConditionClass import SVMCondition
from svm_util.SVMDictType import SVMParametersDict


def get_svm_parameters(
    clf: Any, X_train: Complex128[np.ndarray, "n_samples n_features"]
) -> SVMParametersDict:
    """
    Save SVM parameters for one-vs-rest classification.

    :param clf: scikit-learn SVM object
    :param X_train: Training data
    :return: Dictionary containing SVM parameters (support_vector_array, alpha_array, b_array, n_support_vector)
    """
    support_indices = clf.support_
    support_vector_array = X_train[support_indices]
    alpha_array = clf.dual_coef_[0]
    b_array = clf.intercept_

    data: SVMParametersDict = {
        "support_vector_array": support_vector_array,
        "alpha_array": alpha_array,
        "b_array": b_array,
        "n_support_vector": len(support_vector_array),
    }
    return data


def select_svm_condition():
    """
    Select and configure SVM experimental conditions.

    This function defines various configurations for SVM experiments with different datasets.
    Uncomment the desired configuration to use it.

    Returns:
        SVMCondition: The selected SVM configuration
    """
    """MNISQ MNIST F95, 100 data points per label"""
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=Dataset.MNISQ.Qubits.q10,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #     ds_name=Dataset.MNISQ.Name.MNIST,
    #     fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=list(
    #         range(10)
    #     ),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=100,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """MNISQ MNIST F95, 1000 data points per label"""
    condition = SVMCondition(
        seed=123,
        n_qubits=Dataset.MNISQ.Qubits.q10,
        test_size=0.5,  # Proportion of test data
        C=1.0,  # SVM regularization parameter
        ds_name=Dataset.MNISQ.Name.MNIST,
        fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
        labels=list(
            range(10)
        ),  # Specify labels to load. If None, all labels will be used
        n_data_by_label=1000,  # Number of data points per label. Currently only effective for MNISQ dataset
    )

    """MNISQ Fashion-MNIST F95, 100 data points per label"""
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=Dataset.MNISQ.Qubits.q10,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #
    #     ds_name=Dataset.MNISQ.Name.FashionMNIST,
    #     fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=list(range(10)),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=100,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """MNISQ Fashion-MNIST F95, 1000 data points per label"""
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=Dataset.MNISQ.Qubits.q10,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #
    #     ds_name=Dataset.MNISQ.Name.FashionMNIST,
    #     fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=list(range(10)),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=1000,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """MNISQ Kuzushiji F95, 100 data points per label"""
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=Dataset.MNISQ.Qubits.q10,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #
    #     ds_name=Dataset.MNISQ.Name.Kuzusizi,
    #     fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=list(range(10)),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=100,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """MNISQ Kuzushiji F95, 1000 data points per label"""
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=Dataset.MNISQ.Qubits.q10,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #
    #     ds_name=Dataset.MNISQ.Name.Kuzusizi,
    #     fidelity=Dataset.MNISQ.Fidelity.F95,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=list(range(10)),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=1000,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """VQE-dataset 4qubit"""
    # n_qubits = Dataset.VQEGeneratedDataset.Qubits.q4
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=n_qubits,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #     ds_name=Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset,
    #     fidelity=1.0,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=Dataset.VQEGeneratedDataset.get_hamiltonian_labels(
    #         n_qubits=n_qubits
    #     ),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=300,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """VQE-dataset 8qubit"""
    # n_qubits = Dataset.VQEGeneratedDataset.Qubits.q8
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=n_qubits,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #
    #     ds_name=Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset,
    #     fidelity=1.0,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=Dataset.VQEGeneratedDataset.get_hamiltonian_labels(n_qubits=n_qubits),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=300,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """VQE-dataset 12qubit"""
    # n_qubits = Dataset.VQEGeneratedDataset.Qubits.q12
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=n_qubits,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #     ds_name=Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset,
    #     fidelity=1.0,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=Dataset.VQEGeneratedDataset.get_hamiltonian_labels(
    #         n_qubits=n_qubits
    #     ),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=300,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    """VQE-dataset 16qubit"""
    # n_qubits = Dataset.VQEGeneratedDataset.Qubits.q16
    # condition = SVMCondition(
    #     seed=123,
    #     n_qubits=n_qubits,
    #     test_size=0.5,  # Proportion of test data
    #     C=1.0,  # SVM regularization parameter
    #     ds_name=Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset,
    #     fidelity=1.0,  # Set dataset fidelity. Only effective when MNISQ dataset is selected
    #     labels=Dataset.VQEGeneratedDataset.get_hamiltonian_labels(
    #         n_qubits=n_qubits
    #     ),  # Specify labels to load. If None, all labels will be used
    #     n_data_by_label=300,  # Number of data points per label. Currently only effective for MNISQ dataset
    # )

    return condition


def get_fidelity_kernel_matrix(_x1, _x2):
    """
    Returns a kernel matrix using the fidelity kernel.

    Args:
        _x1: First set of quantum states
        _x2: Second set of quantum states

    Returns:
        Kernel matrix calculated using fidelity between quantum states
    """
    return get_kernel_matrix(x1=_x1, x2=_x2, kernel=fidelity_kernel)
