import os

# Set the number of threads for qulacs
os.environ["QULACS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from dataclasses import asdict

import joblib  # type: ignore
from jax import config

config.update("jax_enable_x64", True)

from aqce_util.AQCEDictType import MatrixCDict
from misc.dataset_loader import load_and_split_dataset_for_additional_learning
from qnn_util.exp_condition_defined import get_defined_qnn_condition
from qnn_util.helper import print_check
from qnn_util.LearningCircuitClass import MyLearningCircuit
from qnn_util.qnn_utility import (
    create_eqs_circuit,
    evaluate_baseline_performance,
    evaluate_performance,
    prepare_experiment_conditions,
    train_qnn,
)


def main(exp_name: str, circuit_type: str, target_label: int) -> None:
    """
    Train a Quantum Neural Network (QNN) using an Eigenvalue Quantum Support (EQS) circuit.

    This function performs the following steps:
    1. Get QNN conditions based on experiment name, circuit type, and target label
    2. Prepare experiment conditions (SVM, AQCE, eigenvalue decomposition)
    3. Load SVM results and get intercepts for each index
    4. Load eigenvalue decomposition results and get eigenvalues
    5. Load quantum circuit optimized with AQCE
    6. Construct EQS circuit using loaded information
    7. Load and split dataset for training and testing
    8. Evaluate baseline performance before additional learning
    9. Train QNN using the EQS circuit
    10. Evaluate model performance after learning
    11. Save results and model data

    Prerequisites:
    - SVM model must be trained (01_train_svm.py)
    - Eigenvalue decomposition must be performed (02_eigenvalue_decompose.py)
    - Quantum circuit EQS must be constructed using Extended AQCE (03_convert_implicit2explicit.py)

    Args:
        exp_name: Name of the experiment
        circuit_type: Type of circuit to use ("random", "predefined", or "random_structure")
        target_label: Target label for binary classification

    Returns:
        None: Results are saved to disk
    """
    qnn_condition = get_defined_qnn_condition(
        exp_name=exp_name, circuit_type=circuit_type, target_label=target_label
    )
    print_check(qnn_condition)
    print(f"circuit_type: {qnn_condition.circuit_type}")

    print(f"Experiment name: {qnn_condition.exp_name}")
    print(f"target_label: {qnn_condition.target_label}")
    print("______________________________________________")

    # 1) First, specify SVMCondition and AQCECondition to load the EQS circuit
    svm_condition, aqce_condition, ed_cfg = prepare_experiment_conditions(
        qnn_condition=qnn_condition
    )

    # 2) Load SVM results and get intercepts for each index
    b_list: list[float] = svm_condition.load_b_list()

    # 3) Load eigenvalue decomposition results and get eigenvalues
    lambda_k_list_list: list[list[float]] = ed_cfg.load_lambda_k_list_list(
        svm_file_name=svm_condition.get_file_name_(), K=qnn_condition.k
    )
    # eigen_datas: list[dict] = load_eigendatas(svm_condition=svm_condition, ed_cfg=ed_cfg) # Load eigendata for each index
    # lambda_k_list_list: list[list[float]] = load_lambda_k_list_list(eigen_datas=eigen_datas, K=qnn_condition.k)

    # 4) Load quantum circuit optimized with AQCE
    path_to_file = aqce_condition.get_save_path(
        svm_filename=svm_condition.get_file_name_(),
        ed_cfg_filename=ed_cfg.get_file_name(),
        today="*",
    )
    matrixC_list: list[MatrixCDict] = aqce_condition.load_matrixC_list(
        path_to_file=path_to_file
    )

    # Construct EQS circuit using loaded information
    circuit: MyLearningCircuit = create_eqs_circuit(qnn_condition, matrixC_list)

    # 5) Load dataset for simulation. Also split it
    # Training data is the same as the training data used in the implicit model. For MNISQ dataset, adjust start_index so that test data is completely new.
    # For VQE dataset, since there is no additional data, use test data only for gradient calculation.
    (
        X_train_original,
        X_test_original,
        y_train_original,
        y_test_original,
        X_train,
        X_test,
        y_train,
        y_test,
    ) = load_and_split_dataset_for_additional_learning(
        qnn_condition=qnn_condition, svm_condition=svm_condition
    )

    # Check for debugging
    print(f"Performance in case of data used for learning the implicit model:")
    _, _ = evaluate_baseline_performance(
        qnn_condition=qnn_condition,
        matrixC_list=matrixC_list,
        lambda_k_list_list=lambda_k_list_list,
        b_list=b_list,
        X_train=X_train_original,
        y_train=y_train_original,
        X_test=X_test_original,
        y_test=y_test_original,
    )

    # 6) Check baseline accuracy before additional learning
    print(f"In case of new data:")
    acc_score_train_baseline, acc_score_test_baseline = evaluate_baseline_performance(
        qnn_condition=qnn_condition,
        matrixC_list=matrixC_list,
        lambda_k_list_list=lambda_k_list_list,
        b_list=b_list,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    # 6) Train QNN
    qnn, opt_loss, opt_params = train_qnn(
        qnn_condition=qnn_condition,
        circuit=circuit,
        lambda_k_list_list=lambda_k_list_list,
        b_list=b_list,
        X_train=X_train,
        y_train=y_train,
    )

    # 7) Evaluate model after learning
    acc_score_train, acc_score_test, y_pred_train, y_pred_test = evaluate_performance(
        qnn=qnn, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )

    # 8) Save various information together
    data_dict = {
        "opt_params": opt_params,
        "opt_loss": opt_loss,
        "acc_score_train": acc_score_train,
        "acc_score_test": acc_score_test,
        "acc_score_train_baseline": acc_score_train_baseline,
        "acc_score_test_baseline": acc_score_test_baseline,
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }
    if qnn.callback_obj is not None:
        data_from_callback_obj = {
            "theta_history": qnn.callback_obj.theta_history,
            "cost_history": qnn.callback_obj.cost_history,
            "iter_history": qnn.callback_obj.iter_history,
            "grad_history": qnn.callback_obj.grad_norm_history,
            "grad_raw_history": qnn.callback_obj.grads_history,
        }
        data_dict.update(data_from_callback_obj)

    save_data_dict = {
        "data_dict": data_dict,
        "qnn_condition": asdict(qnn_condition),
        "svm_condition": asdict(svm_condition),
        "ed_cfg": asdict(ed_cfg),
        "aqce_condition": asdict(aqce_condition),
    }
    joblib.dump(save_data_dict, qnn_condition.get_save_path())


if __name__ == "__main__":
    """Sequential processing"""
    # circuit_type = "random"
    # circuit_type="random_structure"
    circuit_type = "predefined"
    target_labels = list(range(10))
    for target_label in target_labels:
        exp_name = "mnisq-mnist-001_local"
        main(exp_name=exp_name, circuit_type=circuit_type, target_label=target_label)
