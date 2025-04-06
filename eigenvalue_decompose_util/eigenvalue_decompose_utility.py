"""
Eigenvalue Decomposition Utility Module

This module provides functions for eigenvalue decomposition of quantum states and operators.
It includes utilities for:
1. Finding eigenvalues and eigenvectors of measurement operators
2. Orthonormalizing sets of vectors using Gram-Schmidt method
3. Computing observable matrices
4. Denoising kernel matrices
5. Generating Gram matrices for quantum states
6. Expanding eigenvectors in different bases

These utilities are primarily used for analyzing quantum machine learning models,
particularly Support Vector Machines (SVMs) with quantum kernels.
"""

import cvxpy as cp
import numpy as np
from jaxtyping import Complex128, Float64
from scipy.linalg import eigh  # type: ignore

from eigenvalue_decompose_util.EigenvalueDecomposeConditionClass import (
    EigenValueDecomposeCondition,
)
from misc.util import inner_product_exp
from svm_util.SVMDictType import SVMParametersDict


def get_all_eigenvector(
    meas_op: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    absolute_sort=True,
) -> tuple[
    Float64[np.ndarray, "n_support_vectors"],
    Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
]:
    """
    Find the eigenvalues and eigenvectors of a measurement operator.

    Args:
        meas_op: Measurement operator matrix
        absolute_sort: If True, sort eigenvalues by absolute value; otherwise sort by value

    Returns:
        Tuple of (eigenvalues, eigenvectors) sorted in descending order
    """
    values, vectors = eigh(meas_op)
    if absolute_sort:
        order = np.argsort(np.abs(values))[::-1]
    else:
        order = np.argsort(values)[::-1]
    eigenvalues = values[order]
    vectors2 = vectors.T
    eigenvectors = vectors2[order]

    return eigenvalues, eigenvectors


def compute_meas_op_nonumba(
    alpha_array: Complex128[np.ndarray, "n_support_vectors n_features"],
    X_train: Complex128[np.ndarray, "_ n_features"],
) -> Complex128[np.ndarray, "n_features n_features"]:
    """
    Calculate the measurement operator expressed as a linear combination of training data.

    TODO: The current implementation is O(2^n) and takes a considerable amount of time.
    By implementing it as shown in the EQS paper, it can be reduced to O(M^2),
    which would significantly improve performance.

    Args:
        alpha_array: Array of alpha values from SVM
        X_train: Training data array

    Returns:
        Measurement operator matrix
    """
    meas_op = np.zeros((X_train[0].shape[0], X_train[0].shape[0]), dtype=np.complex128)

    for alpha, state_train in zip(alpha_array, X_train):
        _state: np.ndarray = state_train.reshape([-1, 1])  # Reshape to column vector
        meas_op += alpha * (_state @ _state.conj().T)
    return meas_op


def orthonormalize(
    psi_array: Complex128[np.ndarray, "n_support_vectors n_features"],
    G: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    method: str = "gram_schmidt",
) -> tuple[
    Complex128[np.ndarray, "n_support_vectors n_features"],
    Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
]:
    """
    Orthonormalize a set of vectors using the Gram-Schmidt method.

    Args:
        psi_array: List of numpy arrays from $|\psi_1\rangle$ to $|\psi_M\rangle$.
                  Each $|\psi_i\rangle$ is a normalized complex vector (with elements of type np.complex128).
        G: Gram matrix where $G_{ij}=\langle \psi_i|\psi_j\rangle$
        method: Orthonormalization method to use (currently only "gram_schmidt" is supported)

    Returns:
        Tuple of (e_list, coeff_list) where:
        - e_list: List of orthonormal vectors $|e_1\rangle,|e_2\rangle,\dots,|e_M\rangle$
                 obtained by the Gram-Schmidt method
        - coeff_list: Expansion coefficients for each orthonormal vector

    Note:
        The inner products used here are limited to $\langle \psi_i|\psi_j\rangle$ (i.e., between input $|\psi\rangle$ vectors).
        Inner product calculations with vectors other than $|\psi\rangle$ obtained during the process
        are calculated using their $|\psi\rangle$ expansion coefficients and the Gram matrix $G$.
    """
    M = psi_array.shape[0]

    e_list: list[Complex128[np.ndarray, "n_features"]] = (
        []
    )  # List to store the final orthonormal vectors
    coeff_list: list[Complex128[np.ndarray, "n_support_vectors"]] = (
        []
    )  # Expansion coefficients of each $|e_i\rangle$ in terms of $|\psi\rangle$ (vectors of length M)

    if method == "gram_schmidt":
        """Standard Gram-Schmidt method"""
        for k in range(M):
            # The current $|\psi_k\rangle$ can be represented as a standard basis vector in the psi basis
            # That is, the coefficient vector v is a zero vector of length M with v[k]=1
            v = np.zeros(M, dtype=np.complex128)
            v[k] = 1.0

            # Subtract projections onto each already computed orthonormal vector
            # Each $|e_i\rangle$ has a $|\psi\rangle$ expansion $c^{(i)}$, so
            # $\langle e_i|\psi_k\rangle = \sum_j \overline{c^{(i)}_j} \, G[j,k]$ is calculated
            for c in coeff_list:
                p = np.vdot(
                    c, G[:, k]
                )  # p = <e_i|psi_k>, calculation between coefficients only (classical computation)
                v = v - p * c  # Update coefficients (classical computation only)

            # Here v becomes the $|\psi\rangle$ expansion coefficient of $f_k$
            # The norm is calculated as $ \|f_k\|^2 = v^\dagger G\, v $
            norm_sq = np.vdot(
                v, G @ v
            )  # Coefficient calculation is sufficient, so only classical computation is used
            norm_val = np.sqrt(
                norm_sq.real
            )  # norm should be real. Classical computation only.
            assert not np.isclose(
                norm_val, 0
            ), "Failed to obtain linearly independent vectors."  # TODO This code is not working.
            v_norm = (
                v / norm_val
            )  # Normalized expansion coefficients. Classical computation only.

            # $|e_k\rangle$ is constructed using the expansion coefficients v_norm as
            # $|e_k\rangle=\sum_{j=0}^{M-1} (v_norm)_j |\psi_j\rangle$
            e = sum(
                v_norm[j] * psi_array[j] for j in range(M)
            )  # For convenience, calculate the orthonormal vector. In practice, only the coefficients are known, not the quantum state.
            e_list.append(e)
            # Store the coefficients needed to realize the orthonormal vector. Only these classical data are known.
            coeff_list.append(v_norm)

    else:
        raise RuntimeError("method must be gram_schmidt or modified_gram_schmidt")

    return np.array(e_list), np.array(coeff_list)


def compute_observable_matrix(
    G: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    coeff_array: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    alpha_array: Float64[np.ndarray, "n_support_vectors"],
) -> Complex128[np.ndarray, "n_support_vectors n_support_vectors"]:
    """
    Calculate the M×M matrix of the observable O_{α,𝒟} from the coefficient expansion
    of the orthogonal basis and the weights alpha_m.

    Each orthogonal basis |e_i⟩ is expressed in terms of input states as:
      |e_i⟩ = Σₘ c_{i,m} |ψ_m⟩
    where coeff_array contains the 1D array c_{i} (of length M) for each i.

    The matrix elements of the observable are calculated as:
      [O_{α,𝒟}]_{ij} = Σₘ α_m ⋅ (c_{i,m})* ⋅ c_{j,m}

    Args:
        G: Gram matrix where G_{ij} = ⟨ψ_i|ψ_j⟩
        coeff_array: Array of coefficient vectors (each is a 1D np.array of length M)
        alpha_array: 1D np.array of length M, containing weights α_m for each m

    Returns:
        O: Complex matrix of shape (M, M)
    """
    """The following code is a naive implementation"""
    # M = len(coeff_list)
    # O = np.zeros((M, M), dtype=np.complex128)
    #
    # def calc_inner(i, j):
    #     """Calculate the matrix element at row i, column j"""
    #     result = 0
    #     for k, alpha in enumerate(alpha_list):
    #         sum1 = 0
    #         for m in range(M):
    #             sum1 += coeff_list[i][m].conj()*G[m][k]
    #
    #         sum2 = 0
    #         for m in range(M):
    #             sum2 += coeff_list[j][m]*G[k][m]
    #
    #         result += alpha*sum1*sum2
    #     return result
    #
    #
    # for i in range(M):
    #     for j in range(M):
    #         O[i][j] = calc_inner(i=i, j=j)

    """Processed as matrix operations (by Chat-GPT)"""
    # Convert coefficient list to matrix C (shape: (N, M))
    C = coeff_array
    # X[i,k] = sum_m conjugate(C[i, m]) * G[m, k]
    X = np.dot(np.conjugate(C), G)  # shape: (N, M)
    # Y[j,k] = sum_m C[j, m] * G[k, m] = (C dot G.T)[j,k]
    Y = np.dot(C, G.T)  # shape: (N, M)
    # Multiply each column by weight alpha_array[k] and take the inner product
    # O[i,j] = sum_k (X[i,k] * alpha_array[k] * Y[j,k])
    O = np.dot(X * alpha_array, Y.T)  # (X * alpha_array) is a matrix of shape (N, M)

    return O

    # M = len(coeff_list)
    # O = np.zeros((M, M), dtype=np.complex128)
    # # 各 m について外積を取って加算する方法
    # for m in range(M):
    #     # coeff_list[i][m] (i=0,...,M-1) をひとまとめにする。
    #     # ここでは各 m 番目の成分を集めた列ベクトルを作成します。
    #     col = np.array([coeff_list[i][m] for i in range(M)])  # shape (M,)
    #     # 外積: (col.conjugate() (M,) outer col (M,)) で shape (M,M)
    #     O += alpha[m] * np.outer(col.conjugate(), col)
    # return O


def denoise_kernel(
    G: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    lambda_reg: float = 0.0,
) -> Complex128[np.ndarray, "n_support_vectors n_support_vectors"]:
    """
    Calculate a denoised kernel matrix $K$ from a noisy complex kernel matrix $G$
    by solving the following optimization problem:

      minimize    0.5 * ||G - K||_F^2 + lambda_reg * ||K||_*
      subject to  K >> 0,
                  |K[i,j]| <= 1   (for all i,j)

    where $K$ is a Hermitian complex matrix ($K=K^*$).

    Parameters:
        G (numpy.ndarray): Complex kernel matrix with noise
        lambda_reg (float): Regularization parameter

    Returns:
        numpy.ndarray: Denoised kernel matrix
    """
    n = G.shape[0]
    # Define complex Hermitian matrix variable
    K = cp.Variable((n, n), hermitian=True)
    # Add constraints: positive semidefinite and absolute value of each element ≤ 1
    constraints = [K >> 0, cp.abs(K) <= 1]

    # Error term using Frobenius norm (can be calculated for complex matrices)
    frob_term = cp.norm(G - K, "fro") ** 2
    # Nuclear norm (sum of singular values)
    nuc_norm = cp.normNuc(K)

    # Formulation of the objective function
    objective = cp.Minimize(0.5 * frob_term + lambda_reg * nuc_norm)
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem with the SCS solver
    problem.solve(solver=cp.SCS, verbose=True)

    denoised_G = K.value
    if denoised_G is None:
        raise RuntimeError("denoise kernel failed")

    return denoised_G


def generate_G(
    psi_array: Complex128[np.ndarray, "n_support_vectors n_features"],
    noisy: bool,
    n_shots: int | None = None,
    set_diag: bool = True,
    symmetric: bool = True,
) -> Complex128[np.ndarray, "n_support_vectors n_support_vectors"]:
    """
    Function to generate the kernel matrix G.

    Args:
        psi_array: List of quantum states $|psi_i\rangle$
        noisy: Set to True to include noise
        n_shots: Number of measurement shots (only needed if noisy=True)
        set_diag: Set to True to set all diagonal elements to 1 (used as known information)
        symmetric: Set to True to make G[j,i] = G[i,j] by calculating only the upper triangular part

    Returns:
        Kernel matrix (complex type)
    """
    M = psi_array.shape[0]
    G = np.empty((M, M), dtype=np.complex128)

    if symmetric:
        # Calculate only the upper triangular part and set diagonal elements according to options
        for i in range(M):
            for j in range(i, M):
                if i == j:
                    if set_diag:
                        G[i, i] = 1.0
                    else:
                        G[i, i] = inner_product_exp(
                            psi_array[i], psi_array[i], noisy=noisy, n_shots=n_shots
                        )
                else:
                    # Calculate the inner product of psi_array[i] and psi_array[j], and set G[j,i] using symmetry
                    val = inner_product_exp(
                        psi_array[i], psi_array[j], noisy=noisy, n_shots=n_shots
                    )
                    G[i, j] = val
                    G[j, i] = np.conjugate(val)
    else:
        # Calculate inner products for all combinations
        for i in range(M):
            for j in range(M):
                if i == j and set_diag:
                    G[i, j] = 1.0
                else:
                    G[i, j] = inner_product_exp(
                        psi_array[i], psi_array[j], noisy=noisy, n_shots=n_shots
                    )

    return G


def expand_eigenvector(
    compact_eigenvectors: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    coef_array: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
    psi_array: Complex128[np.ndarray, "n_support_vectors n_features"],
) -> tuple[
    Complex128[np.ndarray, "n_support_vectors n_features"],
    Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
]:
    eigen_coeff_array_list: list = []  # Expansion coefficients of $|\psi\rangle$ (vector of length M)
    _eigenvectors = []

    for now_index, eigen_compact in enumerate(compact_eigenvectors):
        print(f"{now_index}/{len(compact_eigenvectors)}")

        eigen_coeff_list: list = []  # Coefficients for data expansion of eigenvectors
        _eigenvector = np.zeros_like(psi_array[0], dtype=np.complex128)
        for i in range(len(psi_array)):
            new_coef = 0
            for j in range(len(eigen_compact)):
                new_coef += eigen_compact[j] * coef_array[j][i]

            eigen_coeff_list.append(new_coef)
            _eigenvector += new_coef * psi_array[i]

        _eigenvectors.append(_eigenvector)
        eigen_coeff_array_list.append(np.array(eigen_coeff_list))
    return np.array(_eigenvectors), np.array(eigen_coeff_array_list)


def delete_nan_value(
    e_array: Complex128[np.ndarray, "n_support_vectors n_features"],
    coef_array: Complex128[np.ndarray, "n_support_vectors n_support_vectors"],
) -> tuple[
    Complex128[np.ndarray, "_ n_features"],
    Complex128[np.ndarray, "_ n_support_vectors"],
]:
    # Keep only elements before any element containing NaN appears
    truncated_e_list = []
    truncated_coef_list = []
    for e_, coef_ in zip(e_array, coef_array):
        if np.isnan(e_).any():
            # If NaN is included, terminate the loop here and remove subsequent elements
            break
        truncated_e_list.append(e_)
        truncated_coef_list.append(coef_)
    return np.array(truncated_e_list, dtype=np.complex128), np.array(
        truncated_coef_list, dtype=np.complex128
    )


def util_get_eigenvector_from_data(
    data: SVMParametersDict,
    method: str,
    noisy: bool,
    n_shots: int | None,
    set_diag: bool,
    symmetric: bool,
    denoise: bool,
    lambda_reg: float,
) -> dict:
    """
    Calculate eigenvectors from SVM data and training data.

    Args:
        data: Dictionary containing SVM parameters
        method: Orthonormalization method to use
        noisy: Whether to simulate measurement noise
        n_shots: Number of measurement shots for noise simulation
        set_diag: Whether to set diagonal elements of the Gram matrix to 1
        symmetric: Whether to make the Gram matrix symmetric
        denoise: Whether to apply denoising to the Gram matrix
        lambda_reg: Regularization parameter for denoising

    Returns:
        Dictionary containing eigenvalues, eigenvectors, and other related data
    """
    psi_array: Complex128[np.ndarray, "n_support_vectors n_features"] = data[
        "support_vector_array"
    ]
    alpha_array: Float64[np.ndarray, "n_support_vectors"] = data["alpha_array"]

    # (1) Calculate the kernel matrix
    G = generate_G(
        psi_array=psi_array,
        noisy=noisy,
        n_shots=n_shots,
        set_diag=set_diag,
        symmetric=symmetric,
    )
    if denoise:
        # Apply correction
        """After verification, this method is sensitive to small numerical errors, so denoise=True is necessary even when there's no noise."""
        G = denoise_kernel(G=G, lambda_reg=lambda_reg)

    # (2) Find orthonormal vectors in the subspace
    e_array, coef_array = orthonormalize(psi_array=psi_array, G=G, method=method)
    # e_array, coef_array = delete_nan_value(e_array, coef_array)

    # (3) Calculate the representation matrix of the observable
    matrix = compute_observable_matrix(
        G=G, coeff_array=coef_array, alpha_array=alpha_array
    )

    # (4) Diagonalize the matrix
    eigenvalues, compact_eigenvectors = get_all_eigenvector(meas_op=matrix)

    # (5) Convert back to the original eigenvectors
    eigenvectors, eigen_coef_list = expand_eigenvector(
        compact_eigenvectors=compact_eigenvectors,
        coef_array=coef_array,
        psi_array=psi_array,
    )

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "compact_eigenvectors": compact_eigenvectors,
        "e_array": e_array,
        "coef_array": coef_array,
        "psi_array": psi_array,
        "alpha_array": alpha_array,
        "G": G,
        "matrix": matrix,
        "eigen_coef_list": eigen_coef_list,
    }


def select_eigenvalue_decompose_condition(
    noisy: bool, n_shots: int | None
) -> EigenValueDecomposeCondition:
    """
    Select and configure eigenvalue decomposition conditions.

    Args:
        noisy: Whether to simulate measurement noise
        n_shots: Number of measurement shots for noise simulation

    Returns:
        EigenValueDecomposeCondition: Configuration for eigenvalue decomposition
    """
    condition = EigenValueDecomposeCondition(
        seed=123,
        # method="modified_gram_schmidt_iterative",
        method="gram_schmidt",
        noisy=noisy,
        n_shots=n_shots,
        set_diag=True,
        symmetric=True,
        denoise=True,
        lambda_reg=0.0,
    )

    return condition
