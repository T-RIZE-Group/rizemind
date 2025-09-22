import os
from pathlib import Path
from typing import Unpack

from eth_account.signers.base import BaseAccount
from hexbytes import HexBytes
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.base_contract import (
    BaseContract,
    FromAddressKwargs,
    contract_factory,
)
from rizemind.contracts.has_account import HasAccount
from web3.contract import Contract

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class TaskAssignmentConfig:
    """Configuration for task assignment."""

    def __init__(self, N: int, T: int, R: int):
        self.N = N  # number of nodes (indices 0..N-1)
        self.T = T  # number of tasks (indices 0..T-1)
        self.R = R  # tasks per node (>=1)

    def to_tuple(self) -> tuple[int, int, int]:
        """Convert to tuple for contract calls."""
        return (self.N, self.T, self.R)


class TaskAssignment(HasAccount, BaseContract):
    """TaskAssignment contract class for minimal-write affine assignment of T tasks to N nodes."""

    def __init__(self, contract: Contract, account: BaseAccount | None = None):
        HasAccount.__init__(self, account=account)
        BaseContract.__init__(self, contract=contract)

    @staticmethod
    def from_address(
        account: BaseAccount | None, **kwargs: Unpack[FromAddressKwargs]
    ) -> "TaskAssignment":
        return TaskAssignment(contract_factory(**kwargs, abi=abi), account=account)

    def initialize(self) -> HexBytes:
        """Initialize the contract. This function can only be called once during proxy deployment."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.initialize(),
            from_account=account,
        )

    def cfg(self, round_id: int) -> TaskAssignmentConfig:
        """Get the configuration for a specific round."""
        result = self.contract.functions.cfg(round_id).call()
        return TaskAssignmentConfig(N=result[0], T=result[1], R=result[2])

    def set_config(self, round_id: int, config: TaskAssignmentConfig) -> HexBytes:
        """Update config for a specific round. Ensure coprimality and ranges."""
        account = self.get_account()
        return self.send(
            tx_fn=self.contract.functions.setConfig(round_id, config.to_tuple()),
            from_account=account,
        )

    def tasks_of_node(self, round_id: int, node_id: int) -> list[int]:
        """Tasks assigned to node n (length = R).

        Args:
            round_id: The round ID
            node_id: Node ID (must be < N)

        Returns:
            List of task IDs assigned to the node
        """
        result = self.contract.functions.tasksOfNode(round_id, node_id).call()
        return list(result)

    def nth_task_of_node(self, round_id: int, node_id: int, task_index: int) -> int:
        """Get the nth task assigned to a node (0-indexed).

        Args:
            round_id: The round ID
            node_id: Node ID (must be < N)
            task_index: Task index (must be < R)

        Returns:
            The task ID
        """
        return self.contract.functions.nthTaskOfNode(
            round_id, node_id, task_index
        ).call()

    def node_count_of_task(self, round_id: int, task_id: int) -> int:
        """Count of nodes assigned to task t.

        Args:
            round_id: The round ID
            task_id: Task ID (must be < T)

        Returns:
            Number of nodes assigned to the task
        """
        return self.contract.functions.nodeCountOfTask(round_id, task_id).call()

    def is_assigned(self, round_id: int, node_id: int, task_id: int) -> bool:
        """Check membership: is node n assigned to task t?

        Args:
            round_id: The round ID
            node_id: Node ID (must be < N)
            task_id: Task ID (must be < T)

        Returns:
            True if the node is assigned to the task
        """
        return self.contract.functions.isAssigned(round_id, node_id, task_id).call()
