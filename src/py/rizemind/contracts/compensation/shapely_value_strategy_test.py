from math import factorial
from rizemind.contracts.compensation.shapely_value_strategy import ShapelyValueStrategy

TEST_CASES = [
    {
        "cs": [(0b00, 90), (0b01, 92), (0b10, 93), (0b11, 90)],
        "player_x_outcome": ([0b01, -0.5], [0b10, 0.5]),
    },
    {
        "cs": [(0b00, 0), (0b01, 0), (0b10, 0), (0b11, 1500)],
        "player_x_outcome": ([0b01, 750], [0b10, 750]),
    },
    {
        "cs": [(0b00, 0), (0b01, 1000), (0b10, 1000), (0b11, 2000)],
        "player_x_outcome": ([0b01, 1000], [0b10, 1000]),
    },
    {
        "cs": [(0b00, 0), (0b01, 0), (0b10, 0), (0b11, 100)],
        "player_x_outcome": ([0b01, 50], [0b10, 50]),
    },
    {
        "cs": [(0b00, 0), (0b01, 100), (0b10, 0), (0b11, 100)],
        "player_x_outcome": ([0b01, 100], [0b10, 0]),
    },
    {
        "cs": [(0b00, 0), (0b01, 1000000), (0b10, 200000), (0b11, 1400000)],
        "player_x_outcome": ([0b01, 1100000], [0b10, 300000]),
    },
    {
        "cs": [(0b00, 0), (0b01, 1000000), (0b10, 500000), (0b11, 1250000)],
        "player_x_outcome": ([0b01, 875000], [0b10, 375000]),
    },
    {
        "cs": [(i, 0 if i == 0 else -100) for i in range(2**7)],
        "player_x_outcome": (
            [0b0000001, -14.285714285714],
            [0b0000010, -14.285714285714],
            [0b0000100, -14.285714285714],
            [0b0001000, -14.285714285714],
            [0b0010000, -14.285714285714],
            [0b0100000, -14.285714285714],
            [0b1000000, -14.285714285714],
        ),
    },
    {
        "cs": [(0b00, 0), (0b01, 3600000), (0b10, 600000), (0b11, 3500000)],
        "player_x_outcome": ([0b01, 3250000], [0b10, 250000]),
    },
    {
        "cs": [(0b00, 0), (0b01, 3600000), (0b10, 600000), (0b11, 3500000)],
        "player_x_outcome": ([0b01, 3250000], [0b10, 250000]),
    },
    {
        "cs": [
            (
                coalition,
                3000000
                if coalition == 0b1111110
                else 3500000
                if coalition == 0b1111111
                else 0,
            )
            for coalition in range(2**7)
        ],
        "player_x_outcome": (
            [0b0000001, 71428.571428571],  # NGO (player 1)
            [0b0000010, 571428.57142857],  # Gov subagency (player 2)
            [0b0000100, 571428.57142857],  # Gov subagency (player 3)
            [0b0001000, 571428.57142857],  # Gov subagency (player 4)
            [0b0010000, 571428.57142857],  # Gov subagency (player 5)
            [0b0100000, 571428.57142857],  # Gov subagency (player 6)
            [0b1000000, 571428.57142857],  # Gov subagency (player 7)
        ),
    },
]


def compute_contributions(player, cs) -> float:
    """
    Calculate the Shapley value for a single player using the correct Shapley formula.

    :param player_bit: The bit representation of the player.
    :param outcomes: A list of tuples where each tuple consists of a coalition (bitmask) and its value.
    :return: The Shapley value of the given player.
    """
    value_dict = dict(cs)
    num_players = bin(max(value_dict.keys())).count(
        "1"
    )  # Count bits in the largest coalition

    shapley = 0

    # Iterate over all possible coalitions excluding the player
    for coalition, value in cs:
        if coalition & player == 0:  # Player is not in the coalition
            new_coalition = coalition | player  # Add player to the coalition
            marginal_contribution = value_dict[new_coalition] - value
            s = bin(coalition).count("1")  # Size of coalition
            shapley += (
                factorial(s) * factorial(num_players - s - 1) * marginal_contribution
            )

    return shapley / factorial(num_players)


def test_compute_contributions():
    # svs = ShapelyValueStrategy(None, None)  # type: ignore
    for test_case in TEST_CASES:
        cs = test_case["cs"]
        for player_x_outcome in test_case["player_x_outcome"]:
            player = player_x_outcome[0]
            expected_contribution = player_x_outcome[1]
            calculated_contribution = compute_contributions(player, cs)
            assert round(expected_contribution, 6) == round(calculated_contribution, 6)
