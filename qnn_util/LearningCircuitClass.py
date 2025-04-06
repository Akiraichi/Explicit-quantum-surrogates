from __future__ import annotations

from dataclasses import dataclass
from typing import List

from qnn_util.helper import FirstKHelper
from qnn_util.skqulacs_circuit import LearningCircuit  # type: ignore


@dataclass(eq=False)
class MyLearningCircuit(LearningCircuit):
    def __post_init__(self):
        super().__post_init__()  # Call the parent class's __post_init__ method to perform initialization

    def backprop_inner_product(self, state, exclude_first_k=False) -> List[float]:
        """
        backprop(self, x: List[float], state) -> List[Float]

        Performs backpropagation using inner product.

        Args:
            state: Quantum state for inner product calculation
            exclude_first_k: If True, excludes the first k parameters

        Returns:
            List of gradient values
        """
        # TODO: This approach may not always work, but it's fine for the implementation used in the current simulation
        # This was done because the for loop processing was wasteful and a cause of slow execution speed. It would have been better to use ndarray processing
        # self._set_input(x)
        ret = self._circuit.backprop_inner_product(state)
        # ans = [0.0] * len(self._learning_parameter_list)
        # for parameter in self._learning_parameter_list:
        #     if not parameter.is_input:
        #         for pos in parameter.positions_in_circuit:
        #             ans[parameter.parameter_id] += ret[pos.gate_pos] * (pos.coef or 1.0)
        if exclude_first_k:
            # Return parameters excluding the first FirstKHelper.param_count parameters. This means returning only the parameters that are optimization targets.
            ret = ret[FirstKHelper.param_count :]

        return ret

    def set_parameters4k_state(self, theta: list[float]) -> None:
        """
        Update the first 3*n_qubits learning parameters with the specified theta values.

        Args:
            theta: List of parameter values to set
        """
        # Initialize the first RX gate parameters to 0.0, effectively making them identity gates.
        _parameters = self.get_parameters()
        assert (
            len(theta) == FirstKHelper.param_count
        ), "This differs from the situation expected by the current implementation"

        for i, t in enumerate(theta):
            # self.circuit._circuit.set_parameter(index=i, parameter=0.0)  # Make it an identity gate
            _parameters[i] = t
        self.update_parameters(theta=_parameters)

    def get_parameters(self, exclude_first_k=False) -> List[float]:
        """Get a list of learning parameters' values."""
        theta_list = [p.value for p in self._learning_parameter_list]
        if exclude_first_k:
            # Return parameters excluding the first FirstKHelper.param_count parameters. This means returning only the parameters that are optimization targets.
            theta_list = theta_list[FirstKHelper.param_count :]
        return theta_list

    def update_parameters(self, theta: List[float]) -> None:
        """Update learning parameter of the circuit with given `theta`.

        Args:
            theta: New learning parameters.
        """

        # Check if it meets the assumptions of the current implementation
        assert (
            len(theta) == self._circuit.get_parameter_count()
        ), "Please check the number of parameters in update_parameters"
        for parameter in self._learning_parameter_list:
            parameter_value = theta[parameter.parameter_id]
            parameter.value = parameter_value
            for pos in parameter.positions_in_circuit:
                self._circuit.set_parameter(
                    pos.gate_pos,
                    parameter_value * (pos.coef or 1.0),
                )
