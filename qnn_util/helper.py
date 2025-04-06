import argparse
import dataclasses

from qnn_util.QNNConditionClass import QNNCondition


@dataclasses.dataclass(frozen=True)
class FirstKHelper:
    param_count: int = 30


def parse_argument():
    parser = argparse.ArgumentParser(description='Run SVM experiment with given parameters.')

    parser.add_argument('--target_label', type=int, required=True, help='Target label')
    parser.add_argument('--exp_number', type=str, required=True, help='Experiment number')
    parser.add_argument('--circuit_type', type=str, required=True, help='circuit type')
    parser.add_argument('--n_data_by_label', type=int, required=True, help='n_data_by_label')
    parser.add_argument('--fidelity', type=float, required=True, help='fidelity')
    parser.add_argument('--k', type=int, required=True, help='n_eigenvector')
    parser.add_argument('--ds_name', type=str, required=True, help='dataset_name')

    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--bwd_parallel', type=str, required=True, help='Backward parallel (True/False)')
    parser.add_argument('--fwd_parallel', type=str, required=True, help='Forward parallel (True/False)')
    parser.add_argument('--bwd_chunk_size', type=int, required=True, help='Backward chunk size')
    parser.add_argument('--fwd_chunk_size', type=int, required=True, help='Forward chunk size')
    parser.add_argument('--n_jobs', type=int, required=True, help='Number of jobs')

    parser.add_argument('--balanced', type=str, required=True, help='Balanced (True/False)')
    parser.add_argument('--use_log_loss', type=str, required=True, help='use_log_loss (True/False)')
    parser.add_argument('--use_hinge_loss', type=str, required=True, help='use_hinge_loss (True/False)')

    parser.add_argument('--maxiter', type=int, required=True, help='Maximum iterations')
    parser.add_argument('--solver_name', type=str, required=True, help='Solver name')
    parser.add_argument('--start_learning_rate', type=float, required=True, help='Starting learning rate')
    parser.add_argument('--b1', type=float, required=True, help='Beta1 for Adam optimizer')
    parser.add_argument('--b2', type=float, required=True, help='Beta2 for Adam optimizer')
    parser.add_argument('--epsilon', type=float, required=True, help='Epsilon for Adam optimizer')
    parser.add_argument('--eps_root', type=float, required=True, help='Epsilon root for Adam optimizer')

    args = parser.parse_args()

    # Convert string arguments to boolean
    args.balanced = args.balanced.lower() == 'true'
    args.use_log_loss = args.use_log_loss.lower() == 'true'
    args.use_hinge_loss = args.use_hinge_loss.lower() == 'true'

    args.bwd_parallel = args.bwd_parallel.lower() == 'true'
    args.fwd_parallel = args.fwd_parallel.lower() == 'true'
    return args


def print_check(qnn_condition: QNNCondition):
    # print(f"Running experiment {args.exp_number} with the following parameters:")
    print(f"circuit_type: {qnn_condition.circuit_type}")

    print(f"fidelity: {qnn_condition.fidelity}")
    print(f"n_data_by_label: {qnn_condition.n_data_by_label}")
    print(f"k: {qnn_condition.k}")
    print(f"ds_name: {qnn_condition.ds_name}")

    print(f"solver_name: {qnn_condition.solver_name}")
    print(f"target_label: {qnn_condition.target_label}")
    print(f"batch_size: {qnn_condition.batch_size}")
    print(f"maxiter: {qnn_condition.maxiter}")
    print(f"balanced: {qnn_condition.balanced}")
    print(f"bwd_parallel: {qnn_condition.bwd_parallel}")
    print(f"fwd_parallel: {qnn_condition.fwd_parallel}")
    print(f"bwd_chunk_size: {qnn_condition.bwd_chunk_size}")
    print(f"fwd_chunk_size: {qnn_condition.fwd_chunk_size}")
    print(f"n_jobs: {qnn_condition.n_jobs}")
    print(f"start_learning_rate: {qnn_condition.start_learning_rate}")
    print(f"b1: {qnn_condition.b1}")
    print(f"b2: {qnn_condition.b2}")
    print(f"epsilon: {qnn_condition.epsilon}")
    print(f"eps_root: {qnn_condition.eps_root}")
