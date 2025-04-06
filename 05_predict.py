import os

# Set the number of threads for qulacs
os.environ["QULACS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import joblib  # type: ignore
from jax import config

config.update("jax_enable_x64", True)
import numpy as np
from jax import numpy as jnp
from jaxtyping import Float64, Int
from sklearn.metrics import accuracy_score  # type: ignore

from aqce_util.AQCEConditionClass import AQCECondition
from aqce_util.AQCEDictType import MatrixCDict
from eigenvalue_decompose_util.EigenvalueDecomposeConditionClass import (
    EigenValueDecomposeCondition,
)
from qnn_util.cost_func_util import update_circuit_from_k_induced_params
from qnn_util.exp_condition_defined import get_defined_qnn_condition
from qnn_util.LearningCircuitClass import MyLearningCircuit
from qnn_util.qnn import MyQNNClassifier
from qnn_util.qnn_utility import (
    create_eqs_circuit,
)
from svm_util.SVMConditionClass import SVMCondition


def main_predict(
    exp_name: str,
    circuit_type: str,
    target_labels: list[int],
    data_type: str,
    use_trained_parameter: bool = True,
) -> tuple[Int[np.ndarray, "n_samples"], Int[np.ndarray, "n_samples"]]:
    """
    Things to do beforehand:
    1. SVM
    2. Creation of observables and eigenvalue decomposition
    3. Construct quantum circuit EQS using Extended AQCE
    4. QNN using EQS
    5. Now here: Prediction using additionally trained EQS

    """
    svm_decision_vector_list: list = []
    for target in target_labels:
        # 1) Load QNN results
        qnn_condition = get_defined_qnn_condition(
            exp_name=exp_name, circuit_type=circuit_type, target_label=target
        )
        data: dict = qnn_condition.load_saved_data(add_path="")
        data_dict: dict = data["data_dict"]
        svm_condition = SVMCondition(**data["svm_condition"])
        aqce_condition = AQCECondition(**data["aqce_condition"])
        ed_cfg = EigenValueDecomposeCondition(**data["ed_cfg"])

        # 2) Load SVM results and get intercepts for each index
        b_list: list[float] = svm_condition.load_b_list()

        # 3) Load eigenvalue decomposition results and get eigenvalues
        lambda_k_list_list: list[list[float]] = ed_cfg.load_lambda_k_list_list(
            svm_file_name=svm_condition.get_file_name_(), K=qnn_condition.k
        )

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

        # Apply parameters after optimization?
        if use_trained_parameter:
            trained_parameter: Float64[np.ndarray, "len_theta"] = data_dict[
                "opt_params"
            ]
            circuit = update_circuit_from_k_induced_params(
                theta_jax=jnp.array(trained_parameter), U=circuit, k=0
            )  # Since this function performs special processing, specify k=0 and handle it

        qnn = MyQNNClassifier(
            circuit=circuit,
            lambda_k_list_list=lambda_k_list_list,
            b_list=b_list,
            qnn_condition=qnn_condition,
            callback_obj=None,
        )

        # Load data used for training
        X_train_original, X_test_original, y_train_original, y_test_original = (
            svm_condition.load_dataset()
        )

        # Make predictions on training data
        if data_type == "train":
            svm_decision_vector = qnn.return_svm_decision_vector(x=X_train_original)
            y_original = y_train_original
        elif data_type == "test":
            svm_decision_vector = qnn.return_svm_decision_vector(x=X_test_original)
            y_original = y_test_original
        else:
            raise NotImplementedError

        svm_decision_vector_list.append(svm_decision_vector)

    svm_decision_two_dim_vector: Float64[np.ndarray, "target n_samples"] = np.array(
        svm_decision_vector_list
    )
    y_predict: Int[np.ndarray, "n_samples"] = np.argmax(
        svm_decision_two_dim_vector, axis=0
    )
    return y_predict, y_original


if __name__ == "__main__":
    exp_name = "mnisq-mnist-001_studio"
    target_labels = list(range(10))

    y_predict, y_original = main_predict(
        exp_name=exp_name,
        # circuit_type="predefined",
        # circuit_type="random",
        circuit_type="random_structure",
        target_labels=target_labels,
        data_type="test",
        # use_trained_parameter=True,
        use_trained_parameter=False,
    )
    acc = accuracy_score(y_original, y_predict)
    print(f"Accuracy after additional learning: {acc}")
