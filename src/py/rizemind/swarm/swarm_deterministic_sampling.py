from rizemind.strategies.contribution.sampling.typing import (
    RandomDeterministicInterface,
)
from rizemind.swarm.swarm import Swarm


class SwarmDeterministicSampling(RandomDeterministicInterface):
    def __init__(self, swarm: Swarm):
        self.swarm = swarm

    def get_number_of_sets(self, round_id: int) -> int:
        number_of_players = self.swarm.trainer_registry.get_trainer_count(round_id)
        return self.swarm.contribution_calculator.get_evaluations_required(
            round_id, number_of_players
        )

    def get_number_of_participants(self, round_id: int) -> int:
        return self.swarm.trainer_registry.get_trainer_count(round_id)

    def get_nth_participant(self, round_id: int, n: int) -> str:
        address = self.swarm.trainer_registry.get_trainer_address_by_id(round_id, n + 1)
        if address is None:
            raise ValueError(
                f"Trainer address not found for round {round_id} and index {n}"
            )
        return address

    def get_nth_set(self, round_id: int, n: int) -> str:
        number_of_players = self.swarm.trainer_registry.get_trainer_count(round_id)
        return str(
            self.swarm.contribution_calculator.get_mask(round_id, n, number_of_players)
        )
