// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.20;

/// @title DemoParams
/// @notice Centralized, easily-tunable parameters for the configurable training demo
/// @dev Edit these constants to quickly change demo behavior without touching the main script
library DemoParams {
    // --- Accounts generation (vm.addr seeds) ---
    uint256 public constant AGGREGATOR_KEY = 1;
    uint256 public constant TRAINER_START_KEY = 2; // trainers will be: TRAINER_START_KEY .. TRAINER_START_KEY + NUM_INITIAL_TRAINERS - 1
    uint256 public constant ADDITIONAL_TRAINER_START_KEY = 100; // additional trainers start
    uint256 public constant EVALUATOR_START_KEY = 200; // evaluators start

    // --- Team sizes ---
    uint256 public constant NUM_INITIAL_TRAINERS = 10;
    uint256 public constant NUM_EVALUATORS = 2;
    uint256 public constant NUM_ADDITIONAL_TRAINERS = 2; // trainers added between rounds

    // --- Scaling Test Configuration ---
    uint256 public constant MIN_TRAINERS = 2;
    uint256 public constant MAX_TRAINERS = 20;
    uint256 public constant TRAINER_STEP_SIZE = 1; // Test every 5th trainer count for efficiency

    // --- Phase configuration ---
    uint256 public constant TRAINING_TTL = 3600; // seconds
    uint256 public constant EVALUATION_TTL = 3600; // seconds
    uint256 public constant EVALUATION_REGISTRATION_TTL = 1800; // seconds

    // --- Contribution Calculator configuration ---
    address public constant CALCULATOR_FEE_RECIPIENT = address(0); // set to nonzero if needed
    uint8 public constant CALCULATOR_INITIAL_WEIGHT = 2;
    uint256 public constant EVALUATIONS_REQUIRED = 16;

    // --- Shapley Sampling Configuration ---
    bool public constant USE_ADAPTIVE_SAMPLING = true; // If true, use 2^n samples; if false, use fixed samples
    uint256 public constant FIXED_SAMPLES_PER_ROUND = 32; // Used when USE_ADAPTIVE_SAMPLING is false

    // --- Selector params ---
    // AlwaysSampled has no params; RandomSampling needs initial stake
    uint256 public constant RANDOM_SAMPLING_INITIAL_STAKE = 1 ether;

    // --- Compensation (ERC20) configuration ---
    string public constant TOKEN_NAME = "DemoToken";
    string public constant TOKEN_SYMBOL = "DEMO";
    uint256 public constant TOKEN_MINT_AMOUNT = 1000 ether;

    // --- Scenario toggles ---
    bool public constant SIMULATE_TRAINER_REMOVAL_IN_ROUND3 = true;
    uint256 public constant TRAINERS_TO_REMOVE = 1; // how many trainers to skip/"remove" in round 3
}
