import sys

from qnn_util.LearningCircuitClass import MyLearningCircuit

sys.path.append("/home/akimoto/mimi/")

import time

import joblib  # type: ignore
import numpy as np
from jaxtyping import Complex128, Float64, Int64
from sklearn.metrics import accuracy_score  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore

from aqce_util.AQCEConditionClass import AQCECondition
from aqce_util.AQCEDictType import MatrixCDict
from eigenvalue_decompose_util.EigenvalueDecomposeConditionClass import (
    EigenValueDecomposeCondition,
)
from qnn_util.aqce_predefined_circuit import (
    aqce_circuit_pretrained,
    aqce_circuit_random_parameter,
    aqce_structure_random_circuit,
)
from qnn_util.OptimizerCallbackClass import OptimizerCallback
from qnn_util.qnn import MyQNNClassifier
from qnn_util.QNNConditionClass import QNNCondition
from svm_util.SVMConditionClass import SVMCondition


def util_predict_check(
    qnn_condition: QNNCondition,
    matrixC_list: list[MatrixCDict],
    lambda_k_list_list: list[list[float]],
    b_list: list[float],
    X_train: Complex128[np.ndarray, "n_train_samples n_features"],
    y_train: Int64[np.ndarray, "n_train_samples"],
    X_test: Complex128[np.ndarray, "n_test_samples n_features"],
    y_test: Int64[np.ndarray, "n_test_samples"],
):
    """Temporary function created for checking prediction accuracy."""
    # Make predictions using the current circuit
    circuit = create_eqs_circuit(qnn_condition=qnn_condition, matrixC_list=matrixC_list)
    print("util_predict_check", qnn_condition.circuit_type)

    _qnn = MyQNNClassifier(
        circuit=circuit,
        lambda_k_list_list=lambda_k_list_list,
        b_list=b_list,
        qnn_condition=qnn_condition,
        callback_obj=None,
    )
    y_pred_train = _qnn.predict(X_train)
    acc_score_train = accuracy_score(y_train, y_pred_train)
    print(f"predict_check_training data accuracy before learning: {acc_score_train}")

    y_pred_test = _qnn.predict(X_test)
    acc_score_test = accuracy_score(y_test, y_pred_test)
    print(f"predict_check_test data accuracy before learning: {acc_score_test}")
    return acc_score_train, acc_score_test


def prepare_experiment_conditions(
    qnn_condition: QNNCondition,
) -> tuple[SVMCondition, AQCECondition, EigenValueDecomposeCondition]:
    """
    Specify SVM execution conditions and AQCE execution conditions based on experiment parameters.

    This function creates condition objects for SVM, AQCE, and eigenvalue decomposition
    based on the provided QNN condition. These objects are used to retrieve saved data.

    Parameters
    ----------
    qnn_condition : QNNCondition
        QNN condition object containing experiment parameters

    Returns
    -------
    tuple[SVMCondition, AQCECondition, EigenValueDecomposeCondition]
        Tuple containing condition objects for SVM, AQCE, and eigenvalue decomposition
    """
    svm_condition = SVMCondition(
        seed=qnn_condition.seed,
        n_qubits=qnn_condition.n_qubits,
        test_size=qnn_condition.test_size,  # Proportion of test data
        C=1.0,  # SVM regularization parameter
        ds_name=qnn_condition.ds_name,
        fidelity=qnn_condition.fidelity,
        labels=qnn_condition.svm_trained_label,
        n_data_by_label=qnn_condition.n_data_by_label,
    )

    aqce_condition = AQCECondition(
        n_qubits=qnn_condition.n_qubits,
        k=qnn_condition.k,
        M_0=12,
        M_max=100000,
        M_delta=6,
        N=100,  # sweep count
        Max_fidelity=0.6,
        data_index=qnn_condition.target_label,
        optimize_method_str="extended_aqce",
        print_debug=True,
        n_data_by_label=qnn_condition.n_data_by_label,
        noisy=False,
        n_shots=None,
    )

    ed_cfg = EigenValueDecomposeCondition(
        seed=123,
        method="gram_schmidt",
        noisy=False,
        n_shots=None,
        set_diag=True,
        symmetric=True,
        denoise=True,
        lambda_reg=0.0,
    )

    return svm_condition, aqce_condition, ed_cfg


def create_eqs_circuit(
    qnn_condition: QNNCondition,
    matrixC_list: list[MatrixCDict],  # TODO: Use proper class name instead of Any.
) -> MyLearningCircuit:
    """
    Create a quantum circuit (EQS) according to the specified circuit_type.

    Args:
        qnn_condition: QNN condition object containing circuit configuration
        matrixC_list: List of matrix C dictionaries for pre-trained gates

    Returns:
        MyLearningCircuit: The constructed quantum circuit
    """

    if qnn_condition.circuit_type == "random":
        circuit = aqce_circuit_random_parameter(
            n_qubits=qnn_condition.n_qubits, pre_trained_gate_list=matrixC_list
        )
    elif qnn_condition.circuit_type == "predefined":
        circuit = aqce_circuit_pretrained(
            n_qubits=qnn_condition.n_qubits, pre_trained_gate_list=matrixC_list
        )
    elif qnn_condition.circuit_type == "random_structure":
        circuit = aqce_structure_random_circuit(
            n_qubits=qnn_condition.n_qubits, pre_trained_gate_list=matrixC_list
        )
    else:
        raise ValueError(f"Unknown circuit type: {qnn_condition.circuit_type}")

    print(f"Number of parameters: {len(circuit.get_parameters())}")
    return circuit


def train_qnn(
    qnn_condition: QNNCondition,
    circuit: MyLearningCircuit,
    lambda_k_list_list: list[list[float]],
    b_list: list[float],
    X_train: Complex128[np.ndarray, "n_train_samples n_features"],
    y_train: Int64[np.ndarray, "n_train_samples"],
) -> tuple[MyQNNClassifier, Float64[np.ndarray, ""], Float64[np.ndarray, "len_theta"]]:
    """
    Initialize and train a QNNClassifier.

    Parameters
    ----------
    qnn_condition : QNNCondition
        Class containing QNN learning conditions.
    circuit : MyLearningCircuit
        Quantum circuit object.
    lambda_k_list_list : List[List[float]]
        List of eigenvalues from SVM.
    b_list : List[float]
        Intercept information from SVM.
    X_train, y_train : ndarray
        Training data.

    Returns
    -------
    qnn : MyQNNClassifier
        Trained (or partially trained) QNN classifier.
    opt_loss : float
        Final cost function value.
    opt_params : ndarray
        Optimized parameters.
    """
    callback = OptimizerCallback(qnn_condition=qnn_condition)
    qnn = MyQNNClassifier(
        circuit=circuit,
        lambda_k_list_list=lambda_k_list_list,
        b_list=b_list,
        qnn_condition=qnn_condition,
        callback_obj=callback,
    )

    start_time = time.time()
    opt_loss, opt_params = qnn.fit(X_train, y_train, maxiter=qnn_condition.maxiter)
    elapsed_time = time.time() - start_time
    print("trained parameters", opt_params)
    print("loss", opt_loss)
    print("learn_time", elapsed_time)

    return qnn, opt_loss, opt_params


def evaluate_baseline_performance(
    qnn_condition: QNNCondition,
    matrixC_list: list[MatrixCDict],
    lambda_k_list_list: list[list[float]],
    b_list: list[float],
    X_train: Complex128[np.ndarray, "n_train_samples n_features"],
    y_train: Int64[np.ndarray, "n_train_samples"],
    X_test: Complex128[np.ndarray, "n_test_samples n_features"],
    y_test: Int64[np.ndarray, "n_test_samples"],
) -> tuple[float, float]:
    """
    Measure baseline prediction accuracy using only existing circuit parameters before additional learning.

    Parameters
    ----------
    qnn_condition : QNNCondition
        QNN condition object.
    matrixC_list : list[MatrixCDict]
        List of matrix C dictionaries for pre-trained gates.
    lambda_k_list_list : list[list[float]]
        List of eigenvalues from SVM.
    b_list : list[float]
        Intercept information from SVM.
    X_train : ndarray
        Training feature data.
    y_train : ndarray
        Training label data.
    X_test : ndarray
        Test feature data.
    y_test : ndarray
        Test label data.

    Returns
    -------
    (acc_score_train_baseline, acc_score_test_baseline) : Tuple[float, float]
        Accuracy scores for training and test data respectively.
    """
    # Calculate prediction accuracy of the quantum circuit before additional learning.
    # Note that this may use different training data than what was used during SVM training.
    # acc_score_train_baseline, acc_score_test_baseline = None, None
    acc_score_train_baseline, acc_score_test_baseline = util_predict_check(
        qnn_condition,
        matrixC_list,
        lambda_k_list_list,
        b_list,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    return acc_score_train_baseline, acc_score_test_baseline


def evaluate_performance(
    qnn: MyQNNClassifier,
    X_train: Complex128[np.ndarray, "n_train_samples n_features"],
    y_train: Int64[np.ndarray, "n_train_samples"],
    X_test: Complex128[np.ndarray, "n_test_samples n_features"],
    y_test: Int64[np.ndarray, "n_test_samples"],
) -> tuple[
    float,
    float,
    Int64[np.ndarray, "n_train_samples"],
    Int64[np.ndarray, "n_test_samples"],
]:
    """
    Calculate accuracy for training and test data using the trained model.

    Parameters
    ----------
    qnn : MyQNNClassifier
        Trained QNN classifier.
    X_train : ndarray
        Training feature data.
    y_train : ndarray
        Training label data.
    X_test : ndarray
        Test feature data.
    y_test : ndarray
        Test label data.

    Returns
    -------
    (acc_score_train, acc_score_test, y_pred_train, y_pred_test)
        Training accuracy, test accuracy, and predicted labels for each dataset.
    """
    y_pred_train = qnn.predict(X_train)
    acc_score_train = accuracy_score(y_train, y_pred_train)

    start_time = time.time()
    y_pred_test = qnn.predict(X_test)
    acc_score_test = accuracy_score(y_test, y_pred_test)
    predict_time = time.time() - start_time
    print(f"Test data prediction time: {predict_time}")
    print(f"Training data accuracy after additional learning: {acc_score_train}")
    print(f"Test data accuracy after additional learning: {acc_score_test}")

    return acc_score_train, acc_score_test, y_pred_train, y_pred_test
