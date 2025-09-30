from eth_typing import ChecksumAddress


class ParticipantMapping:
    """Mapping between trainer addresses and their ids.

    This class maintains a bijective mapping between trainer addresses and integer IDs,
    enabling efficient bit-mask representation of trainer coalitions for Shapley value
    calculation.

    Attributes:
        participant_ids: Dictionary mapping trainer addresses to their unique IDs.
    """

    participant_ids: dict[ChecksumAddress, int]

    def __init__(self) -> None:
        """Initialize an empty participant mapping."""
        self.participant_ids = {}

    def add_participant(self, participant: ChecksumAddress) -> None:
        """Add a participant to the mapping.

        If the participant doesn't already exist, assigns them the next available ID.

        Args:
            participant: The trainer address to add.
        """
        if participant not in self.participant_ids:
            self.participant_ids[participant] = self.get_total_participants()

    def get_total_participants(self) -> int:
        """Get the total number of participants.

        Returns:
            The count of unique participants in the mapping.
        """
        return len(self.participant_ids.values())

    def get_participant_id(self, participant: ChecksumAddress) -> int:
        """Get the numerical ID for a participant.

        Args:
            participant: The trainer address to look up.

        Returns:
            The numerical ID assigned to this participant.

        Raises:
            ValueError: If the participant is not in the mapping.
        """
        if participant not in self.participant_ids:
            raise ValueError(f"{participant} did not participate.")
        return self.participant_ids[participant]

    def get_participant_mask(self, participant: ChecksumAddress) -> int:
        """Get the bit mask for a participant.

        Args:
            participant: The trainer address.

        Returns:
            An integer with a single bit set representing this participant.
        """
        participant_id = self.get_participant_id(participant)
        return 1 << participant_id

    def get_participant_set_id(self, participants: list[ChecksumAddress]) -> str:
        """Generate a unique set ID for a group of participants.

        Args:
            participants: List of trainer addresses in the set.

        Returns:
            String representation of the bit mask identifying this set.
        """
        return self.include_participants(participants=participants, id="0")

    def in_set(self, trainer: ChecksumAddress, id: str) -> bool:
        """Check if a trainer is a member of a set.

        Args:
            trainer: The trainer address to check.
            id: The set identifier (bit mask as string).

        Returns:
            True if the trainer is in the set, False otherwise.
        """
        aggregate_mask = int(id)
        trainer_mask = self.get_participant_mask(trainer)
        return (aggregate_mask & trainer_mask) != 0

    def exclude_participants(
        self,
        participants: ChecksumAddress | list[ChecksumAddress],
        id: str | None = None,
    ):
        """Remove participants from a set.

        Args:
            participants: Single trainer address or list of addresses to exclude.
            id: The set identifier to modify. If None, starts with empty set.

        Returns:
            String representation of the updated set bit mask.
        """
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
        """Add participants to a set.

        Args:
            participants: Single trainer address or list of addresses to include.
            id: The set identifier to modify. If None, starts with empty set.

        Returns:
            String representation of the updated set bit mask.
        """
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
        """Get all participant addresses.

        Returns:
            List of all trainer addresses in the mapping.
        """
        return list(self.participant_ids.keys())
