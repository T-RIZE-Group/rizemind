from eth_typing import ChecksumAddress


class ParticipantMapping:
    participant_ids: dict[ChecksumAddress, int]

    def __init__(self) -> None:
        self.participant_ids = {}

    def add_participant(self, participant: ChecksumAddress) -> None:
        if participant not in self.participant_ids:
            self.participant_ids[participant] = self.get_total_participants()

    def get_total_participants(self) -> int:
        return len(self.participant_ids.values())

    def get_participant_id(self, participant: ChecksumAddress) -> int:
        if participant not in self.participant_ids:
            raise ValueError(f"{participant} did not participate")
        return self.participant_ids[participant]

    def get_participant_mask(self, participant: ChecksumAddress) -> int:
        participant_id = self.get_participant_id(participant)
        return 1 << participant_id

    def get_participant_set_id(self, participants: list[ChecksumAddress]) -> str:
        return self.include_participants(participants=participants, id="0")

    def in_set(self, trainer: ChecksumAddress, id: str) -> bool:
        aggregate_mask = int(id)
        trainer_mask = self.get_participant_mask(trainer)
        return (aggregate_mask & trainer_mask) != 0

    def exclude_participants(
        self,
        participants: ChecksumAddress | list[ChecksumAddress],
        id: str | None = None,
    ):
        aggregate_mask = int(id) if id is not None else 0
        if isinstance(participants, list):
            for participant in participants:
                participant_mask = self.get_participant_mask(participant)
                aggregate_mask &= ~participant_mask
        else:
            participant_mask = self.get_participant_mask(participants)
            aggregate_mask &= ~participant_mask
        return str(aggregate_mask)

    def include_participants(
        self,
        participants: ChecksumAddress | list[ChecksumAddress],
        id: str | None = None,
    ):
        aggregate_mask = int(id) if id is not None else 0
        if isinstance(participants, list):
            for participant in participants:
                participant_mask = self.get_participant_mask(participant)
                aggregate_mask |= participant_mask
        else:
            participant_mask = self.get_participant_mask(participants)
            aggregate_mask |= participant_mask
        return str(aggregate_mask)

    def get_participants(self) -> list[ChecksumAddress]:
        return list(self.participant_ids.keys())
