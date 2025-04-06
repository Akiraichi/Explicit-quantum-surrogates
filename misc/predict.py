from typing import List

import numpy as np
from jaxtyping import Complex128, Int
from qulacs import QuantumGateMatrix, QuantumState

from aqce_util.AQCEConditionClass import AQCECondition
from eigenvalue_decompose_util.EigenvalueDecomposeConditionClass import (
    EigenValueDecomposeCondition,
)
from misc.kernel import fidelity_kernel
from misc.util import (
    calc_fidelity_between_state_and_k_state,
    update_state_from_dagger_gate_matrix_list,
)
from svm_util.SVMConditionClass import SVMCondition


def predict_one_data_using_theoretical_eqs(
    x: np.ndarray,
    eigenvector_k_list_list: List[List[np.ndarray]],
    lambda_k_list_list: List[List[float]],
    b_list,
):
    scores = []
    for eigenvector_k_list, b, lambda_k_list in zip(
        eigenvector_k_list_list, b_list, lambda_k_list_list
    ):
        score = 0
        for i, (eigenvector, lamda) in enumerate(
            zip(eigenvector_k_list, lambda_k_list)
        ):
            score += lamda * np.abs(np.vdot(eigenvector, x)) ** 2
        score += b
        scores.append(score)

    return np.argmax(scores)  # Return the index of the class with the highest score


def predict_using_theoretical_eqs(
    _k: int, svm_conditiion: SVMCondition, ed_cfg: EigenValueDecomposeCondition, X_test
):
    """Predict using the exact eigenvector model of EQS (Eigenvalue Quantum Support)."""

    # 1) Load eigenvalues of the observable from the implicit model
    _lambda_k_list_list: list[list[float]] = ed_cfg.load_lambda_k_list_list(
        svm_file_name=svm_conditiion.get_file_name_(), K=_k
    )
    # _lambda_k_list_list: list[list[float]] = load_lambda_k_list_list(eigen_datas=eigen_datas, K=_k)
    # Get the intercept values
    b_list = svm_conditiion.load_b_list()

    # 2) Construct the exact eigenvector model
    # Get _k eigenvectors for each data_index
    eigenvector_k_list_list = ed_cfg.load_eigenvector_k_list_list(
        svm_file_name=svm_conditiion.get_file_name_(), K=_k
    )
    # eigenvector_k_list_list = load_eigenvector_k_list_list(eigen_datas=eigen_datas, K=_k)

    # 3) Predict for each data point
    result_theoretical_eqs = []
    for i, _x in enumerate(X_test):
        result = predict_one_data_using_theoretical_eqs(
            x=_x,
            eigenvector_k_list_list=eigenvector_k_list_list,
            lambda_k_list_list=_lambda_k_list_list,
            b_list=b_list,
        )
        result_theoretical_eqs.append(result)
    return result_theoretical_eqs


def predict_using_eqs(
    X_test: Complex128[np.ndarray, "n_samples n_features"],
    data_index_list: list[int],
    aqce_condition: AQCECondition,
    svm_condition: SVMCondition,
    ed_cfg: EigenValueDecomposeCondition,
    K: int,
    noisy: bool,
    n_shots: int | None,
) -> Int[np.ndarray, "n_samples"]:
    """Predict using EQS and return the prediction results.

    Specifically:
    (1) Construct EQS using quantum circuits built with AQCE
    (2) Make predictions using EQS

    eigen_datas: eigen_data for each data_index stored in data_index order.
    """

    # 1) Load eigenvalues of the observable from the implicit model
    lambda_k_list_list: list[list[float]] = ed_cfg.load_lambda_k_list_list(
        svm_file_name=svm_condition.get_file_name_(), K=K
    )

    # 2) Load quantum circuits constructed with AQCE
    C_dagger_list: list[list[QuantumGateMatrix]] = []
    for data_index in data_index_list:
        print("data_index:", data_index)
        path_to_file = aqce_condition.search_aqce_data_path_to_file(
            data_index=data_index,
            svm_filename=svm_condition.get_file_name_(),
            ed_cfg_filename=ed_cfg.get_file_name(),
        )
        C_dagger: list[QuantumGateMatrix] = aqce_condition.load_AQCE_circuit_C_dagger(
            path_to_file=path_to_file
        )
        C_dagger_list.append(C_dagger)

    # 3) Predict for each data point using EQS
    y_predict_list: list = []
    for index_x, _x in enumerate(X_test):
        print("index_x:", index_x)
        # Predict using EQS. Calculate |state>=C^†|x> to compute |<k| C^†|x>|^2.
        # Calculate |state>=C^†|x> to compute |<k| C^†|x>|^2.
        state_list = []
        for C_dagger in C_dagger_list:
            x = QuantumState(qubit_count=aqce_condition.n_qubits)
            x.load(_x.tolist())
            _state = update_state_from_dagger_gate_matrix_list(
                state=x, C_dagger=C_dagger
            )
            state = _state.get_vector()
            state_list.append(state)

        # EQS回路の出力量子状態をもとに、予測を実行する
        y_predict: int = predict_one_data_using_eqs_from_output_state_list(
            output_state_list=state_list,
            lambda_k_list_list=lambda_k_list_list,
            K=K,
            b_list=svm_condition.load_b_list(),
            noisy=noisy,
            n_shots=n_shots,
        )
        #
        y_predict_list.append(y_predict)

    return np.asarray(y_predict_list)


def predict_one_data_using_eqs_from_output_state_list(
    output_state_list: list[Complex128[np.ndarray, "n_features"]],
    lambda_k_list_list: List[List[float]],
    K: int,
    b_list: list[float],
    noisy=False,
    n_shots: int | None = None,
) -> int:
    """Execute prediction based on the output quantum states of the EQS circuit."""
    scores: list[float] = []
    for state, b, lambda_k_list in zip(output_state_list, b_list, lambda_k_list_list):
        # Execute binary classification for each target
        score = 0.0
        for i, lamda in enumerate(lambda_k_list):
            # λ*|<state|k>|^2. Since we take the absolute value, the order of the inner product doesn't matter.
            score += lamda * calc_fidelity_between_state_and_k_state(
                state=state, k=i, K=K, noisy=noisy, n_shots=n_shots
            )
        score += b
        scores.append(score)

    return int(np.argmax(scores))  # Return the index of the class with the highest score


def predict_using_kernel_svm(
    X: Complex128[np.ndarray, "n_samples n_features"],
    svm_datas: list,
    noisy=False,
    n_shots: int | None = None,
) -> Int[np.ndarray, "n_samples"]:
    """Perform multi-class classification using a trained SVM with quantum kernel."""
    y_predict = []
    for _x in X:
        scores = []
        for svm_data in svm_datas:
            score = sum(
                svm_data["alpha_array"][i]
                * fidelity_kernel(
                    x=svm_data["support_vector_array"][i],
                    y=_x,
                    noisy=noisy,
                    n_shots=n_shots,
                )
                for i in range(len(svm_data["support_vector_array"]))
            )
            score += svm_data["b_array"][0]
            scores.append(score)
        y_predict.append(np.argmax(scores))
    return np.asarray(y_predict, dtype=np.int64)
