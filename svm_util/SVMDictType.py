from typing import Any, Dict, List, TypedDict

import numpy as np
from jaxtyping import Complex128, Float64, Int64


class SVMParametersDict(TypedDict):
    """
    Type definition for SVM parameters dictionary.

    This dictionary stores the essential parameters of a trained SVM model,
    including support vectors, alpha coefficients, bias term, and the number
    of support vectors.

    Attributes:
        support_vector_array: Array of support vectors
        alpha_array: Array of alpha coefficients for each support vector
        b_array: Bias term of the decision function
        n_support_vector: Number of support vectors
    """

    support_vector_array: Complex128[np.ndarray, "n_support_vectors n_features"]
    alpha_array: Float64[np.ndarray, "n_support_vectors"]
    b_array: Float64[np.ndarray, ""]
    n_support_vector: int


class SVMSaveDataDict(TypedDict):
    """
    Type definition for SVM save data dictionary.

    This dictionary stores all data related to an SVM experiment, including the
    configuration, trained models, datasets, and evaluation results. It is used
    for saving and loading SVM experiment results.

    Attributes:
        condition: Dictionary of SVM experiment conditions
        datas: List of SVM parameter dictionaries, one for each binary classifier
        X: Complete feature dataset
        y: Complete label dataset
        X_train: Training feature data
        X_test: Test feature data
        y_train: Training label data
        y_test: Test label data
        acc_train: Accuracy on training data
        acc_test: Accuracy on test data
        predict_train: Predicted labels for training data
        predict_test: Predicted labels for test data
    """

    condition: Dict[str, Any]
    datas: List[SVMParametersDict]
    X: Complex128[np.ndarray, "n_samples n_features"]
    y: Int64[np.ndarray, "n_samples"]
    X_train: Complex128[np.ndarray, "_n_train_samples n_features"]
    X_test: Complex128[np.ndarray, "_n_test_samples n_features"]
    y_train: Int64[np.ndarray, "_n_train_samples"]
    y_test: Int64[np.ndarray, "_n_test_samples"]
    acc_train: float
    acc_test: float
    predict_train: Int64[np.ndarray, "_n_train_samples"]
    predict_test: Int64[np.ndarray, "_n_test_samples"]
