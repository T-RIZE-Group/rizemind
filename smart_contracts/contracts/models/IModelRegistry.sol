struct RoundSummary {
    uint256 roundId;
    uint64 nTrainers;
    uint64 modelScore;
    uint128 totalContributions;
}

interface IModelRegistry {
    function canTrain(address trainer, uint256 roundId) external returns (bool);
    function curentRound() external view returns (uint256);
    function nextRound(RoundSummary calldata summary) external;
}
