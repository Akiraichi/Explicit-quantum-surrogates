"""
Implicit to Explicit Model Conversion Script

This script converts implicit quantum machine learning models to explicit quantum circuits
using the AQCE. The conversion process involves:

1. Loading an implicit model (SVM) trained on quantum data
2. Loading eigenvalue decomposition data for the model
3. Setting up AQCE execution conditions
4. Running the AQCE algorithm to find quantum circuits that represent the model's eigenvectors
5. Saving the resulting explicit model for later use

The script supports both a newer implementation and a legacy implementation (via the old_expriment flag).
It can also simulate measurement noise for more realistic results.
"""

import os

os.environ["QULACS_NUM_THREADS"] = (
    "1"  # There are cases where setting the number of threads to 1 is faster
)
os.chdir(os.path.abspath("/Users/nakayama/PycharmProjects/aoba"))

from qulacs import QuantumState

from aqce_util.aqce_util import select_aqce_condition
from aqce_util.extended_aqce import ExtendedAQCE
from eigenvalue_decompose_util.eigenvalue_decompose_utility import (
    select_eigenvalue_decompose_condition,
)
from svm_util.svm_utility import select_svm_condition


def process_task(
    k: int,
    data_index: int,
    noisy: bool,
    n_shots: int | None,
    old_expriment: bool = False,
):
    """
    Process a task to convert an implicit model to an explicit quantum circuit.

    This function loads an implicit model (SVM) and its eigenvalue decomposition data,
    then uses AQCE (Adaptive Quantum Circuit Embedding) to find quantum circuits
    that represent the model's eigenvectors.

    Args:
        k: Number of eigenvectors to embed in the explicit model
        data_index: Index of the data to process from the eigenvalue decomposition results
        noisy: Whether to simulate measurement noise
        n_shots: Number of measurement shots for noise simulation (required if noisy=True)
        old_expriment: If True, uses the older implementation method

    Returns:
        None: Results are saved to disk but not returned
    """
    # 1) Specify the setting of the implicit model to load
    svm_condition = select_svm_condition()

    # 2) Also specify the eigenvalue decomposition method
    ed_cfg = select_eigenvalue_decompose_condition(noisy=noisy, n_shots=n_shots)
    # Load data
    eigen_datas = ed_cfg.load_saved_data(svm_file_name=svm_condition.get_file_name_())

    # 3) Set AQCE execution conditions
    aqce_condition = select_aqce_condition(
        n_qubits=svm_condition.n_qubits,
        n_data_by_label=svm_condition.n_data_by_label,
        data_index=data_index,
        k=k,
        noisy=noisy,
        n_shots=n_shots,
    )
    assert aqce_condition.n_qubits == svm_condition.n_qubits

    if not old_expriment:
        # 4) Execute
        # Select data for the index to be processed this time
        data = eigen_datas[data_index]

        psi_list = []
        for psi in data["psi_list"]:
            state_quantum = QuantumState(qubit_count=aqce_condition.n_qubits)
            state_quantum.load(psi)
            psi_list.append(state_quantum)

        target_state_list: list[QuantumState] = []
        for i in range(k):
            # Eigenvector
            state_array = data["eigenvectors"][i]
            state_quantum = QuantumState(qubit_count=aqce_condition.n_qubits)
            state_quantum.load(state_array)
            target_state_list.append(state_quantum)

        aqce = ExtendedAQCE(
            target_state_list=target_state_list,
            condition=aqce_condition,
            coef_ndarray=data["eigen_coef_list"][:k],  # To embed k eigenvectors
            data_list=psi_list,
        )

    else:
        """Past code. Can be used as is"""
        # 3) Execute
        # Select data for the index to be processed this time
        data = eigen_datas[data_index]

        target_state_list = []
        for i in range(k):
            state_array = data["eigenvectors"][i]
            state_quantum = QuantumState(qubit_count=aqce_condition.n_qubits)
            state_quantum.load(state_array)
            target_state_list.append(state_quantum)

        aqce = ExtendedAQCE(
            target_state_list=target_state_list, condition=aqce_condition
        )

    aqce.run()
    # 5) Save
    aqce.save(
        save_path=aqce_condition.get_save_path(
            svm_filename=svm_condition.get_file_name_(),
            ed_cfg_filename=ed_cfg.get_file_name(),
        )
    )


if __name__ == "__main__":
    """When executing with qsub"""
    # k = int(sys.argv[1])  # 1,2,3,4
    # data_index = int(sys.argv[2])  # 0~5
    # process_task(k=k, data_index=data_index)

    """When you want to process sequentially"""
    noisy = True
    n_shots = 100_0000
    n_qubits = 10

    # noisy = False
    # n_shots = None
    # n_qubits = 10
    data_index_list = [0]
    k_list = [2]
    for data_index in data_index_list:
        for k in k_list:
            process_task(k=k, data_index=data_index, noisy=noisy, n_shots=n_shots)

    """When you want to process in parallel using processes"""
    # import concurrent.futures
    #
    # noisy = False
    # n_shots = None
    # n_qubits = 10
    # data_index_list = list(range(10))
    # k_list = list(range(1, 11))
    #
    # with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
    #     futures = [
    #         executor.submit(process_task, k, data_index, noisy, n_shots)
    #         for data_index in data_index_list
    #         for k in k_list
    #     ]
    #
    #     # Wait for all tasks to complete
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             future.result()  # Get the result here (if needed)
    #         except Exception as exc:
    #             print(f"Task generated an exception: {exc}")
