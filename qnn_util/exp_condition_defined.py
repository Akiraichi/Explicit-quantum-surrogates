import argparse

from misc.DatasetClass import Dataset
from qnn_util.errors import ConfigurationError
from qnn_util.QNNConditionClass import QNNCondition


def get_defined_qnn_condition(
    exp_name: str, circuit_type: str, target_label: int
) -> QNNCondition:
    """
    Get a predefined QNN condition based on experiment name, circuit type, and target label.

    This function returns a QNNCondition object with predefined parameters for different
    experiment configurations. Each configuration sets various parameters for the quantum
    neural network, including dataset settings, parallel processing options, cost function
    settings, and optimization parameters.

    Args:
        exp_name: Name of the experiment configuration to use
        circuit_type: Type of circuit to use ("random", "predefined", or "random_structure")
        target_label: Target label for binary classification

    Returns:
        QNNCondition: A configuration object for the quantum neural network

    Raises:
        ConfigurationError: If an invalid experiment name is provided
    """
    # MNISQ-MNIST, 1000 data points, target_label=0
    if exp_name == "mnisq-mnist-000_studio":
        args = argparse.Namespace(
            # Experiment conditions
            target_label=target_label,
            exp_name=exp_name,
            debug_print=True,
            circuit_type=circuit_type,
            # Dataset
            n_qubits=10,
            fidelity=Dataset.MNISQ.Fidelity.F95,
            n_data_by_label=1000,
            svm_trained_label=list(range(10)),
            start_index=1000,
            k=3,
            ds_name=Dataset.MNISQ.Name.MNIST,
            # Parallel processing
            batch_size=100,
            bwd_parallel=True,
            fwd_parallel=True,
            bwd_chunk_size=100,  # Set chunk_size*n_jobs=10000 because K*n_data=10000. Note that when batch processing, the batch size becomes n_data
            fwd_chunk_size=100,
            n_jobs=10,
            # Cost function
            balanced=True,
            use_log_loss=True,
            use_hinge_loss=False,
            # Optimization
            maxiter=10,
            solver_name="adam",
            start_learning_rate=0.009,
            b1=0.9,
            b2=0.999,
            epsilon=1e-8,
            eps_root=1e-8,
        )
    elif exp_name == "mnisq-mnist-011_lg":
        import multiprocessing as mp
        mp.set_start_method('spawn')  # Because os.fork() conflicts with jax

        # Changes are in chunk_size and n_jobs
        args = argparse.Namespace(
            # Experiment conditions
            target_label=target_label,
            exp_name=exp_name,
            debug_print=True,
            circuit_type=circuit_type,
            # Dataset
            n_qubits=10,
            fidelity=Dataset.MNISQ.Fidelity.F95,
            n_data_by_label=1000,
            svm_trained_label=list(range(10)),
            start_index=1000,
            k=10,
            ds_name=Dataset.MNISQ.Name.MNIST,
            # Parallel processing
            batch_size=100,
            bwd_parallel=True,
            fwd_parallel=True,
            bwd_chunk_size=5000,  # Set chunk_size*n_jobs=10000 because K*n_data*0.5*n_label=50000. Note that when batch processing, this becomes the batch size
            fwd_chunk_size=5000,
            n_jobs=10,
            # Cost function
            balanced=True,
            use_log_loss=False,
            use_hinge_loss=True,
            # Optimization
            maxiter=100,
            solver_name="adam",
            start_learning_rate=0.009,
            b1=0.9,
            b2=0.999,
            epsilon=1e-8,
            eps_root=1e-8,
        )
    elif exp_name == "mnisq-mnist-001_studio":
        # Changes from 000: changed to hinge loss
        args = argparse.Namespace(
            # Experiment conditions
            target_label=target_label,
            exp_name=exp_name,
            debug_print=True,
            circuit_type=circuit_type,
            # Dataset
            n_qubits=10,
            fidelity=Dataset.MNISQ.Fidelity.F95,
            n_data_by_label=1000,
            svm_trained_label=list(range(10)),
            start_index=1000,
            k=3,
            ds_name=Dataset.MNISQ.Name.MNIST,
            # Parallel processing
            batch_size=100,
            bwd_parallel=True,
            fwd_parallel=True,
            bwd_chunk_size=100,  # Set chunk_size*n_jobs=10000 because K*n_data=10000. Note that when batch processing, the batch size becomes n_data
            fwd_chunk_size=100,
            n_jobs=6,
            # Cost function
            balanced=True,
            use_log_loss=False,
            use_hinge_loss=True,
            # Optimization
            maxiter=10,
            solver_name="adam",
            start_learning_rate=0.009,
            b1=0.9,
            b2=0.999,
            epsilon=1e-8,
            eps_root=1e-8,
        )

    elif exp_name == "mnisq-mnist-001_local":
        # Changes from 000: changed to hinge loss
        args = argparse.Namespace(
            # Experiment conditions
            target_label=target_label,
            exp_name=exp_name,
            debug_print=True,
            circuit_type=circuit_type,
            # Dataset
            n_qubits=10,
            fidelity=Dataset.MNISQ.Fidelity.F95,
            n_data_by_label=1000,
            svm_trained_label=list(range(10)),
            start_index=1000,
            k=3,
            ds_name=Dataset.MNISQ.Name.MNIST,
            # Parallel processing
            batch_size=100,
            bwd_parallel=True,
            fwd_parallel=True,
            bwd_chunk_size=100,  # Set chunk_size*n_jobs=10000 because K*n_data=10000. Note that when batch processing, the batch size becomes n_data
            fwd_chunk_size=100,
            n_jobs=6,
            # Cost function
            balanced=True,
            use_log_loss=False,
            use_hinge_loss=True,
            # Optimization
            maxiter=10,
            solver_name="adam",
            start_learning_rate=0.009,
            b1=0.9,
            b2=0.999,
            epsilon=1e-8,
            eps_root=1e-8,
        )

    elif exp_name == "vqe-000_studio":
        # VQE dataset, 8qubits, k=6
        args = argparse.Namespace(
            # Experiment conditions
            target_label=target_label,
            exp_name=exp_name,
            debug_print=True,
            # circuit_type="random",
            # circuit_type="random_structure",
            circuit_type=circuit_type,
            # Dataset
            n_qubits=8,
            fidelity=1.0,
            n_data_by_label=300,
            svm_trained_label=list(range(6)),
            start_index=0,
            k=6,
            ds_name=Dataset.VQEGeneratedDataset.Name.VQEGeneratedDataset,
            # Parallel processing
            batch_size=100,
            bwd_parallel=True,
            fwd_parallel=True,
            bwd_chunk_size=100,
            fwd_chunk_size=100,
            n_jobs=6,  # A bit fewer than usual
            # Cost function
            balanced=True,
            use_log_loss=True,
            use_hinge_loss=False,
            # Optimization
            maxiter=10,
            solver_name="adam",
            start_learning_rate=0.009,
            b1=0.9,
            b2=0.999,
            epsilon=1e-8,
            eps_root=1e-8,
        )
    else:
        raise ConfigurationError(f"Invalid configuration name: '{exp_name}' was specified.")

    qnn_condition = QNNCondition(
        # Other settings
        seed=123,
        exp_name=args.exp_name,
        debug_print=args.debug_print,
        n_qubits=args.n_qubits,
        # Experiment conditions
        circuit_type=args.circuit_type,
        k=args.k,  # Number of eigenvectors to embed in EQS
        target_label=args.target_label,  # Target label for learning
        # Dataset
        ds_name=args.ds_name,
        fidelity=args.fidelity,
        n_data_by_label=args.n_data_by_label,
        svm_trained_label=args.svm_trained_label,
        test_size=0.5,
        start_index=args.start_index,
        # Parallel processing
        fwd_parallel=args.fwd_parallel,
        bwd_parallel=args.bwd_parallel,
        fwd_chunk_size=args.fwd_chunk_size,
        bwd_chunk_size=args.bwd_chunk_size,
        n_jobs=args.n_jobs,
        # Cost function
        balanced=args.balanced,
        use_log_loss=args.use_log_loss,
        use_hinge_loss=args.use_hinge_loss,
        # Optimization
        maxiter=args.maxiter,
        solver_name=args.solver_name,
        batch_size=args.batch_size,
        start_learning_rate=args.start_learning_rate,
        b1=args.b1,
        b2=args.b2,
        epsilon=args.epsilon,
        eps_root=args.eps_root,
    )

    return qnn_condition
