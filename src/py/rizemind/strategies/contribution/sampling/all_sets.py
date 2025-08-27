import itertools

from eth_typing import ChecksumAddress
from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy
from rizemind.authentication.authenticated_client_properties import (
    AuthenticatedClientProperties,
)
from rizemind.strategies.contribution.sampling.sets_sampling_strat import (
    SetsSamplingStrategy,
)
from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSet,
)


class AllSets(SetsSamplingStrategy):
    trainer_mapping: ParticipantMapping
    current_round: int
    sets: dict[str, TrainerSet]

    def __init__(self) -> None:
        super().__init__()
        self.trainer_mapping = ParticipantMapping()
        self.current_round = -1  # Initialize to invalid round
        self.sets = {}

    def sample_trainer_sets(
        self, server_round: int, results: list[tuple[ClientProxy, FitRes]]
    ) -> list[TrainerSet]:
        if server_round == self.current_round:
            return self.get_sets(round_id=server_round)

        self.current_round = server_round
        self.trainer_mapping = ParticipantMapping()
        self.sets = {}

        for client, _ in results:
            auth = AuthenticatedClientProperties.from_client(client)
            self.trainer_mapping.add_participant(auth.trainer_address)

        results_coalitions = [
            list(combination)
            for r in range(len(results) + 1)
            for combination in itertools.combinations(results, r)
        ]
        for results_coalition in results_coalitions:
            members: list[ChecksumAddress] = []
            for client, _ in results_coalition:
                auth = AuthenticatedClientProperties.from_client(client)
                members.append(auth.trainer_address)
            id = self.trainer_mapping.get_participant_set_id(members)
            self.sets[id] = TrainerSet(id, members)

        return self.get_sets(round_id=server_round)

    def get_sets(self, round_id: int) -> list[TrainerSet]:
        """Return all trainer sets for the given round."""
        if round_id != self.current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self.current_round}"
            )
        return list(self.sets.values())

    def get_set(self, round_id: int, id: str) -> TrainerSet:
        """Return a specific trainer set by ID for the given round."""
        if round_id != self.current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self.current_round}"
            )
        if id not in self.sets:
            raise ValueError(f"Trainer set with ID {id} not found")
        return self.sets[id]

    def get_trainer_mapping(self, round_id: int) -> ParticipantMapping:
        if round_id != self.current_round:
            raise ValueError(
                f"Round {round_id} is not the current round {self.current_round}"
            )
        return self.trainer_mapping
