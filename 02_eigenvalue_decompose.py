"""
Eigenvalue Decomposition Script

This script performs eigenvalue decomposition on SVM model data. It loads SVM data,
calculates eigenvalues and eigenvectors using specified methods, and optionally
simulates noise in the calculations.

The script supports two different implementation approaches:
1. A newer method that uses util_get_eigenvector_from_data
2. An older method that creates a matrix and diagonalizes it directly

The results can be used for further quantum machine learning analysis.
"""

import joblib  # type: ignore

from eigenvalue_decompose_util.eigenvalue_decompose_utility import (
    compute_meas_op_nonumba,
    get_all_eigenvector,
    select_eigenvalue_decompose_condition,
    util_get_eigenvector_from_data,
)
from eigenvalue_decompose_util.EigenvalueDecomposeDictType import (
    EigenValueDecomposeDataDict,
)
from svm_util.svm_utility import (
    select_svm_condition,
)
from svm_util.SVMConditionClass import SVMCondition
from svm_util.SVMDictType import SVMParametersDict, SVMSaveDataDict


def process_task(noisy: bool, n_shots: int | None, old_expriment: bool = False) -> None:
    """
    Process eigenvalue decomposition tasks on SVM data.

    This function loads SVM data, performs eigenvalue decomposition using the specified
    method, and optionally simulates measurement noise. It can use either a newer
    implementation or an older one based on the old_expriment parameter.

    Args:
        noisy: If True, simulates measurement noise in the calculations
        n_shots: Number of measurement shots for noise simulation (required if noisy=True)
        old_expriment: If True, uses the older implementation method that creates a matrix
                      and diagonalizes it directly

    Returns:
        None: Results are stored in memory but not returned or saved by default
              (saving code is commented out)
    """
    # 1) Specify SVM execution conditions to specify the data to load
    svm_condition: SVMCondition = select_svm_condition()

    # 2) Load data
    svm_data_dict: SVMSaveDataDict = svm_condition.load_saved_data()
    datas: list[SVMParametersDict] = svm_data_dict["datas"]

    # 3) Calculate eigenvalues and eigenvectors
    cfg_ed = select_eigenvalue_decompose_condition(noisy=noisy, n_shots=n_shots)
    datas_: list[EigenValueDecomposeDataDict] = []
    if not old_expriment:
        for i, data in enumerate(datas):
            #
            eigenvector_data = util_get_eigenvector_from_data(
                data=data,
                method=cfg_ed.method,
                noisy=cfg_ed.noisy,
                n_shots=cfg_ed.n_shots,
                set_diag=cfg_ed.set_diag,
                symmetric=cfg_ed.symmetric,
                denoise=cfg_ed.denoise,
                lambda_reg=cfg_ed.lambda_reg,
            )
            data_: EigenValueDecomposeDataDict = {
                "eigenvalues": eigenvector_data["eigenvalues"],
                "eigenvectors": eigenvector_data["eigenvectors"],
                "compact_eigenvectors": eigenvector_data["compact_eigenvectors"],
                "e_array": eigenvector_data["e_array"],
                "coef_array": eigenvector_data["coef_array"],
                "psi_list": eigenvector_data["psi_array"],  # Named _list for compatibility, but it's an array in the current implementation
                "alpha_array": eigenvector_data["alpha_array"],
                "G": eigenvector_data["G"],
                "matrix": eigenvector_data["matrix"],
                "eigen_coef_list": eigenvector_data["eigen_coef_list"],  # Named _list for compatibility, but it's an array in the current implementation
                "condition": cfg_ed,
            }
            datas_.append(data_)
    else:
        """Old implementation. This API is also kept so it can be executed.
        It creates a matrix and diagonalizes it. This may be better in terms of accuracy in some cases.
        """
        for i, data in enumerate(datas):
            print(f"{i}th_start calculation of measurement operator")
            # Calculate the measurement operator expressed as a linear combination of training data
            meas_op = compute_meas_op_nonumba(
                alpha_array=data["alpha_array"], X_train=data["support_vector_array"]
            )
            print(f"{i}th_start eigenvalue decomposition")
            # Find the maximum eigenvalue of the measurement operator and its eigenvector
            _eigenvalues, _eigenvectors = get_all_eigenvector(meas_op=meas_op)
            data_old = {
                "eigenvalues": _eigenvalues,
                "eigenvectors": _eigenvectors,
                "matrix": meas_op,
            }
            datas_.append(data_old)  # type: ignore

        # 4) Save the calculated results
        joblib.dump(
            datas_,
            cfg_ed.get_save_path(svm_file_name=svm_condition.get_file_name_()),
        )


if __name__ == "__main__":
    # No noise
    process_task(noisy=False, n_shots=None)
    # With noise
    # process_task(noisy=True, n_shots=100_0000)
