from typing import Protocol

from eth_typing import ChecksumAddress
from flwr.common import FitRes
from web3 import Web3

from rizemind.strategies.contribution.sampling.sets_sampling_strat import (
    SetsSamplingStrategy,
)
from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import TrainerSet


class RandomDeterministicInterface(Protocol):
    def get_number_of_sets(self, round_id: int) -> int: ...
    def get_number_of_participants(self, round_id: int) -> int: ...
    def get_nth_participant(self, round_id: int, n: int) -> str: ...
    def get_nth_set(self, round_id: int, n: int) -> str: ...


class RandomDeterministicSampling(SetsSamplingStrategy):
    _trainer_mapping: ParticipantMapping
    _sampler: RandomDeterministicInterface
    _current_round: int
    _sets: dict[str, TrainerSet]

    def __init__(self, sampler: RandomDeterministicInterface) -> None:
        self._sampler = sampler
        self._current_round = -1
        self._sets = {}

    def sample_trainer_sets(
        self, server_round: int, results: list[tuple[ChecksumAddress, FitRes]]
    ) -> list[TrainerSet]:
        if server_round == self._current_round:
            return self.get_sets(round_id=server_round)

        self._current_round = server_round
        self._trainer_mapping = ParticipantMapping()
        self._sets = {}

        n_participants = self._sampler.get_number_of_participants(server_round)
        for i in range(n_participants):
            participant = self._sampler.get_nth_participant(server_round, i)
            self._trainer_mapping.add_participant(Web3.to_checksum_address(participant))

        n_sets = self._sampler.get_number_of_sets(server_round)
        """
        For each set, we add the set and the complementary set to calculate Shapley Value
        """
        for i in range(n_sets):
            set_id = self._sampler.get_nth_set(server_round, i)
            members = self._trainer_mapping.get_participants_of_set_id(set_id)
            self._sets[set_id] = TrainerSet(set_id, members, order=i)
            for participant in self._trainer_mapping.get_participants():
                complementary_id = self._trainer_mapping.get_complementary_id(
                    participant, set_id
                )

                complementary_members = (
                    self._trainer_mapping.get_participants_of_set_id(complementary_id)
                )
                if complementary_id not in self._sets:
                    self._sets[complementary_id] = TrainerSet(
                        complementary_id, complementary_members, order=i
                    )

        return self.get_sets(round_id=server_round)

    def get_sets(self, round_id: int) -> list[TrainerSet]:
        """Return all trainer sets for the given round."""
        if round_id != self._current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self._current_round}"
            )
        return list(self._sets.values())

    def get_set(self, round_id: int, id: str) -> TrainerSet:
        """Return a specific trainer set by ID for the given round."""
        if round_id != self._current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self._current_round}"
            )
        if id not in self._sets:
            raise ValueError(f"Trainer set with ID {id} not found")
        return self._sets[id]

    def get_trainer_mapping(self, round_id: int) -> ParticipantMapping:
        if round_id != self._current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self._current_round}"
            )
        return self._trainer_mapping
