import pytest
from eth_account import Account
from eth_typing import ChecksumAddress
from flwr.common.typing import Parameters

from rizemind.strategies.contribution.calculators.shapley_value import (
    ShapleyValueCalculator,
)
from rizemind.strategies.contribution.shapley.trainer_mapping import ParticipantMapping
from rizemind.strategies.contribution.shapley.trainer_set import (
    TrainerSetAggregate,
    TrainerSetAggregateStore,
)


"""
Test suite for the ShapleyValueCalculator class.

This test suite validates the implementation of the Shapley value calculation algorithm
for federated learning contribution assessment. The tests cover:

1. Happy path scenarios with known expected values
2. Edge cases (single participant, empty participants)
3. Custom scoring functions
4. Mathematical properties (efficiency, symmetry)
5. Large coalition scenarios
6. Error handling and robustness

The Shapley value is calculated using the formula:
φ_i = Σ_{S⊆N\\{i}} (|S|!(|N|-|S|-1)! / |N|!) * [v(S∪{i}) - v(S)]

Where:
- φ_i is the Shapley value for player i
- S is a coalition not containing player i
- N is the set of all players
- v(S) is the value of coalition S
- |S| is the size of coalition S
"""


@pytest.fixture
def sample_addresses() -> list[ChecksumAddress]:
    """Generate sample Ethereum addresses for testing."""
    return [
        Account.create().address,
        Account.create().address,
        Account.create().address,
    ]


@pytest.fixture
def participant_mapping(sample_addresses: list[ChecksumAddress]) -> ParticipantMapping:
    """Create a ParticipantMapping with sample addresses."""
    mapping = ParticipantMapping()
    for address in sample_addresses:
        mapping.add_participant(address)
    return mapping


@pytest.fixture
def trainer_set_store(
    sample_addresses: list[ChecksumAddress],
) -> TrainerSetAggregateStore:
    """Create a TrainerSetAggregateStore with sample coalitions."""
    store = TrainerSetAggregateStore()

    # Create empty coalition (no participants)
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 0.0
    store.insert(empty_coalition)

    # Create single participant coalitions
    for i, address in enumerate(sample_addresses):
        coalition = TrainerSetAggregate(
            id=str(1 << i),  # Use bit mask as ID
            members=[address],
            parameters=Parameters([], ""),
            config={},
        )
        coalition.loss = 100.0 + i * 50.0  # Different losses for each participant
        store.insert(coalition)

    # Create coalition with all participants
    all_participants_coalition = TrainerSetAggregate(
        id=str((1 << len(sample_addresses)) - 1),  # All bits set
        members=sample_addresses,
        parameters=Parameters([], ""),
        config={},
    )
    all_participants_coalition.loss = 500.0
    store.insert(all_participants_coalition)

    return store


@pytest.fixture
def shapley_calculator() -> ShapleyValueCalculator:
    """Create a ShapleyValueCalculator instance."""
    return ShapleyValueCalculator()


def test_shapley_value_calculator_with_existing_fixtures(
    shapley_calculator: ShapleyValueCalculator,
    participant_mapping: ParticipantMapping,
    trainer_set_store: TrainerSetAggregateStore,
):
    """Test Shapley value calculation using the existing fixtures."""

    # Calculate Shapley values using the fixture data
    scores = shapley_calculator.get_scores(
        participant_mapping=participant_mapping, store=trainer_set_store
    )

    # Verify results
    assert len(scores) == 3
    assert all(addr in scores for addr in participant_mapping.get_participants())

    # All scores should be non-negative
    assert all(score.score >= 0 for score in scores.values())

    # Sum should be reasonable (may not equal grand coalition due to missing coalitions)
    total_shapley = sum(score.score for score in scores.values())
    assert total_shapley >= 0.0  # Should be non-negative
    assert total_shapley <= 500.0  # Should not exceed grand coalition value

    # Verify PlayerScore structure
    for addr in participant_mapping.get_participants():
        assert scores[addr].trainer_address == addr
        assert isinstance(scores[addr].score, float)


def test_shapley_value_calculator_happy_path(
    shapley_calculator: ShapleyValueCalculator,
    participant_mapping: ParticipantMapping,
    trainer_set_store: TrainerSetAggregateStore,
):
    """Test the happy path for Shapley value calculation with a simple 2-participant scenario."""

    # Create a simpler scenario with just 2 participants for easier verification
    simple_mapping = ParticipantMapping()
    simple_store = TrainerSetAggregateStore()

    # Add 2 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    simple_mapping.add_participant(addr1)
    simple_mapping.add_participant(addr2)

    # Create coalitions with known values for easy Shapley calculation
    # Empty coalition: value = 0
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 0.0
    simple_store.insert(empty_coalition)

    # Coalition with participant 1: value = 100
    coalition_1 = TrainerSetAggregate(
        id="1",  # Binary: 01
        members=[addr1],
        parameters=Parameters([], ""),
        config={},
    )
    coalition_1.loss = 100.0
    simple_store.insert(coalition_1)

    # Coalition with participant 2: value = 200
    coalition_2 = TrainerSetAggregate(
        id="2",  # Binary: 10
        members=[addr2],
        parameters=Parameters([], ""),
        config={},
    )
    coalition_2.loss = 200.0
    simple_store.insert(coalition_2)

    # Coalition with both participants: value = 400
    coalition_both = TrainerSetAggregate(
        id="3",  # Binary: 11
        members=[addr1, addr2],
        parameters=Parameters([], ""),
        config={},
    )
    coalition_both.loss = 400.0
    simple_store.insert(coalition_both)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(
        participant_mapping=simple_mapping, store=simple_store
    )

    # Verify results
    assert len(scores) == 2
    assert addr1 in scores
    assert addr2 in scores

    # For this simple case, we can calculate expected Shapley values:
    # Player 1:
    #   - Marginal contribution when joining empty: 100 - 0 = 100
    #   - Marginal contribution when joining player 2: 400 - 200 = 200
    #   - Shapley value = (100 + 200) / 2 = 150
    # Player 2:
    #   - Marginal contribution when joining empty: 200 - 0 = 200
    #   - Marginal contribution when joining player 1: 400 - 100 = 300
    #   - Shapley value = (200 + 300) / 2 = 250

    assert scores[addr1].score == pytest.approx(150.0, rel=1e-10)
    assert scores[addr2].score == pytest.approx(250.0, rel=1e-10)

    # Verify PlayerScore structure
    assert scores[addr1].trainer_address == addr1
    assert scores[addr2].trainer_address == addr2


def test_shapley_value_calculator_single_participant(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation with a single participant."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add single participant
    addr = Account.create().address
    mapping.add_participant(addr)

    # Create coalitions
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 0.0
    store.insert(empty_coalition)

    single_coalition = TrainerSetAggregate(
        id="1", members=[addr], parameters=Parameters([], ""), config={}
    )
    single_coalition.loss = 100.0
    store.insert(single_coalition)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results
    assert len(scores) == 1
    assert addr in scores
    assert scores[addr].score == pytest.approx(100.0, rel=1e-10)


def test_shapley_value_calculator_empty_participants(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation with no participants."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Create only empty coalition
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 0.0
    store.insert(empty_coalition)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results
    assert len(scores) == 0


def test_shapley_value_calculator_custom_scoring_function(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation with a custom scoring function."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 2 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    mapping.add_participant(addr1)
    mapping.add_participant(addr2)

    # Create coalitions with custom metrics
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.metrics = {"custom_score": 0.0}
    store.insert(empty_coalition)

    coalition_1 = TrainerSetAggregate(
        id="1", members=[addr1], parameters=Parameters([], ""), config={}
    )
    coalition_1.metrics = {"custom_score": 50.0}
    store.insert(coalition_1)

    coalition_2 = TrainerSetAggregate(
        id="2", members=[addr2], parameters=Parameters([], ""), config={}
    )
    coalition_2.metrics = {"custom_score": 75.0}
    store.insert(coalition_2)

    coalition_both = TrainerSetAggregate(
        id="3", members=[addr1, addr2], parameters=Parameters([], ""), config={}
    )
    coalition_both.metrics = {"custom_score": 150.0}
    store.insert(coalition_both)

    # Custom scoring function that uses metrics instead of loss
    def custom_scoring(coalition: TrainerSetAggregate) -> float:
        metric_value = coalition.get_metric("custom_score", 0.0)
        if isinstance(metric_value, (int, float)):
            return float(metric_value)
        return 0.0

    # Calculate Shapley values with custom scoring
    scores = shapley_calculator.get_scores(
        participant_mapping=mapping, store=store, coalition_to_score_fn=custom_scoring
    )

    # Verify results
    assert len(scores) == 2
    assert addr1 in scores
    assert addr2 in scores

    # Expected Shapley values with custom scoring:
    # Player 1: (50 - 0 + 150 - 75) / 2 = 62.5
    # Player 2: (75 - 0 + 150 - 50) / 2 = 87.5

    assert scores[addr1].score == pytest.approx(62.5, rel=1e-10)
    assert scores[addr2].score == pytest.approx(87.5, rel=1e-10)


def test_shapley_value_calculator_three_participants(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation with three participants."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 3 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    addr3 = Account.create().address
    mapping.add_participant(addr1)
    mapping.add_participant(addr2)
    mapping.add_participant(addr3)

    # Create all possible coalitions (2^3 = 8 coalitions)
    coalitions = [
        ("0", [], 0.0),  # Empty
        ("1", [addr1], 10.0),  # Player 1 only
        ("2", [addr2], 20.0),  # Player 2 only
        ("3", [addr1, addr2], 35.0),  # Players 1+2
        ("4", [addr3], 30.0),  # Player 3 only
        ("5", [addr1, addr3], 45.0),  # Players 1+3
        ("6", [addr2, addr3], 55.0),  # Players 2+3
        ("7", [addr1, addr2, addr3], 70.0),  # All players
    ]

    for coalition_id, members, value in coalitions:
        coalition = TrainerSetAggregate(
            id=coalition_id, members=members, parameters=Parameters([], ""), config={}
        )
        coalition.loss = value
        store.insert(coalition)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results
    assert len(scores) == 3
    assert all(addr in scores for addr in [addr1, addr2, addr3])

    # Verify that all scores are non-negative (since all marginal contributions are positive)
    assert all(score.score >= 0 for score in scores.values())

    # Verify that the sum of Shapley values equals the grand coalition value
    total_shapley = sum(score.score for score in scores.values())
    assert total_shapley == pytest.approx(70.0, rel=1e-10)


def test_shapley_value_calculator_missing_coalitions(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation when some coalitions are missing from the store."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 2 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    mapping.add_participant(addr1)
    mapping.add_participant(addr2)

    # Only create some coalitions (missing the empty coalition and coalition with both players)
    coalition_1 = TrainerSetAggregate(
        id="1", members=[addr1], parameters=Parameters([], ""), config={}
    )
    coalition_1.loss = 100.0
    store.insert(coalition_1)

    coalition_2 = TrainerSetAggregate(
        id="2", members=[addr2], parameters=Parameters([], ""), config={}
    )
    coalition_2.loss = 200.0
    store.insert(coalition_2)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results - should still work but with incomplete calculations
    assert len(scores) == 2
    assert addr1 in scores
    assert addr2 in scores

    # Since some coalitions are missing, the Shapley values may not be as expected
    # but the calculation should complete without errors
    assert isinstance(scores[addr1].score, float)
    assert isinstance(scores[addr2].score, float)


def test_shapley_value_calculator_zero_weight_scenario(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation when all weights are zero (edge case)."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 2 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    mapping.add_participant(addr1)
    mapping.add_participant(addr2)

    # Create coalitions where all marginal contributions are zero
    # This can happen when all coalitions have the same value
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 100.0
    store.insert(empty_coalition)

    coalition_1 = TrainerSetAggregate(
        id="1", members=[addr1], parameters=Parameters([], ""), config={}
    )
    coalition_1.loss = 100.0  # Same as empty
    store.insert(coalition_1)

    coalition_2 = TrainerSetAggregate(
        id="2", members=[addr2], parameters=Parameters([], ""), config={}
    )
    coalition_2.loss = 100.0  # Same as empty
    store.insert(coalition_2)

    coalition_both = TrainerSetAggregate(
        id="3", members=[addr1, addr2], parameters=Parameters([], ""), config={}
    )
    coalition_both.loss = 100.0  # Same as all others
    store.insert(coalition_both)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results - all scores should be 0 since no marginal contribution
    assert len(scores) == 2
    assert addr1 in scores
    assert addr2 in scores
    assert scores[addr1].score == pytest.approx(0.0, rel=1e-10)
    assert scores[addr2].score == pytest.approx(0.0, rel=1e-10)


def test_shapley_value_calculator_mathematical_properties(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test that Shapley values satisfy mathematical properties like efficiency and symmetry."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 3 participants
    addr1 = Account.create().address
    addr2 = Account.create().address
    addr3 = Account.create().address
    mapping.add_participant(addr1)
    mapping.add_participant(addr2)
    mapping.add_participant(addr3)

    # Create coalitions with symmetric values
    coalitions = [
        ("0", [], 0.0),  # Empty
        ("1", [addr1], 10.0),  # Player 1 only
        ("2", [addr2], 10.0),  # Player 2 only (same as player 1)
        ("3", [addr1, addr2], 25.0),  # Players 1+2
        ("4", [addr3], 10.0),  # Player 3 only (same as others)
        ("5", [addr1, addr3], 25.0),  # Players 1+3 (same as 1+2)
        ("6", [addr2, addr3], 25.0),  # Players 2+3 (same as others)
        ("7", [addr1, addr2, addr3], 45.0),  # All players
    ]

    for coalition_id, members, value in coalitions:
        coalition = TrainerSetAggregate(
            id=coalition_id, members=members, parameters=Parameters([], ""), config={}
        )
        coalition.loss = value
        store.insert(coalition)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results
    assert len(scores) == 3
    assert all(addr in scores for addr in [addr1, addr2, addr3])

    # Test efficiency: sum of Shapley values should equal grand coalition value
    total_shapley = sum(score.score for score in scores.values())
    assert total_shapley == pytest.approx(45.0, rel=1e-10)

    # Test symmetry: players with identical marginal contributions should have same Shapley values
    # In this symmetric setup, all players should have equal Shapley values
    assert scores[addr1].score == pytest.approx(scores[addr2].score, rel=1e-10)
    assert scores[addr2].score == pytest.approx(scores[addr3].score, rel=1e-10)
    assert scores[addr1].score == pytest.approx(scores[addr3].score, rel=1e-10)

    # Each player should get 1/3 of the total value due to symmetry
    expected_individual_score = 45.0 / 3
    assert scores[addr1].score == pytest.approx(expected_individual_score, rel=1e-10)


def test_shapley_value_calculator_large_coalition(
    shapley_calculator: ShapleyValueCalculator,
):
    """Test Shapley value calculation with a larger number of participants."""

    mapping = ParticipantMapping()
    store = TrainerSetAggregateStore()

    # Add 4 participants
    addresses = [Account.create().address for _ in range(4)]
    for addr in addresses:
        mapping.add_participant(addr)

    # Create coalitions with proper bit mask IDs based on participant IDs
    # Empty coalition
    empty_coalition = TrainerSetAggregate(
        id="0", members=[], parameters=Parameters([], ""), config={}
    )
    empty_coalition.loss = 0.0
    store.insert(empty_coalition)

    # Single participant coalitions - use the actual participant IDs from mapping
    for addr in addresses:
        participant_id = mapping.get_participant_id(addr)
        coalition_id = str(1 << participant_id)
        coalition = TrainerSetAggregate(
            id=coalition_id, members=[addr], parameters=Parameters([], ""), config={}
        )
        coalition.loss = 10.0 * (participant_id + 1)
        store.insert(coalition)

    # Grand coalition - include all participants
    grand_coalition_id = mapping.get_participant_set_id(addresses)
    grand_coalition = TrainerSetAggregate(
        id=grand_coalition_id,
        members=addresses,
        parameters=Parameters([], ""),
        config={},
    )
    grand_coalition.loss = 100.0
    store.insert(grand_coalition)

    # Calculate Shapley values
    scores = shapley_calculator.get_scores(participant_mapping=mapping, store=store)

    # Verify results
    assert len(scores) == 4
    assert all(addr in scores for addr in addresses)

    # All scores should be non-negative
    assert all(score.score >= 0 for score in scores.values())

    # Sum should equal grand coalition value
    total_shapley = sum(score.score for score in scores.values())
    assert total_shapley == pytest.approx(100.0, rel=1e-10)

    # Verify that the calculation completed without errors
    # (The exact values are complex to calculate manually due to factorial weights)
    for addr in addresses:
        assert isinstance(scores[addr].score, float)
        assert not (
            scores[addr].score < 0 or scores[addr].score > 100.0
        )  # Reasonable bounds
