from aqce_util.AQCEConditionClass import AQCECondition


def select_aqce_condition(
    n_qubits: int,
    n_data_by_label: int,
    k: int,
    data_index: int,
    noisy: bool,
    n_shots: int | None,
) -> AQCECondition:
    """
    Select and configure AQCE (Adaptive Quantum Circuit Embedding) execution conditions.

    This function creates an AQCECondition object with conventional settings for the
    AQCE algorithm, which is used to find quantum circuits that represent eigenvectors.

    Args:
        n_qubits: Number of qubits in the quantum system
        n_data_by_label: Number of data points per label
        k: Number of eigenvectors to embed
        data_index: Index of the data to process
        noisy: Whether to simulate measurement noise
        n_shots: Number of measurement shots for noise simulation (required if noisy=True)

    Returns:
        AQCECondition: Configuration object for AQCE execution
    """
    aqce_condition = AQCECondition(
        n_qubits=n_qubits,
        k=k,
        M_0=12,
        M_max=100000,
        M_delta=6,
        N=100,  # sweep count
        Max_fidelity=0.6,
        data_index=data_index,
        optimize_method_str="extended_aqce",
        print_debug=True,
        n_data_by_label=n_data_by_label,
        noisy=noisy,
        n_shots=n_shots,
    )
    return aqce_condition
