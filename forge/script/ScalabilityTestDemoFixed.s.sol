// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.20;

import "forge-std/Script.sol";
import "forge-std/console.sol";

import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "@rizemind-contracts/contribution/CalculatorFactory.sol";
import {AccessControlFactory} from "@rizemind-contracts/access/AccessControlFactory.sol";
import {CompensationFactory} from "@rizemind-contracts/compensation/CompensationFactory.sol";
import {AlwaysSampled} from "@rizemind-contracts/sampling/AlwaysSampled.sol";
import {RandomSampling} from "@rizemind-contracts/sampling/RandomSampling.sol";
import {ContributionCalculator} from "@rizemind-contracts/contribution/ContributionCalculator.sol";
import {BaseAccessControl} from "@rizemind-contracts/access/BaseAccessControl.sol";
import {SimpleMintCompensation} from "@rizemind-contracts/compensation/SimpleMintCompensation.sol";
import {BaseTrainingPhases} from "@rizemind-contracts/training/BaseTrainingPhases.sol";
import {DemoParams} from "./DemoParams.sol";

/// @title ScalabilityTestDemoFixed
/// @notice Fixed version that handles negative Shapley values properly
/// @dev Tests different trainer counts with proper reward calculation
contract ScalabilityTestDemoFixed is Script {
    SwarmV1 public testSwarm;
    BaseAccessControl public accessControl;
    SimpleMintCompensation public compensation;
    ContributionCalculator public contributionCalculator;

    address[] public trainers;
    address[] public evaluators;
    address public aggregator;

    // Data collection for graphing and analysis
    struct TestResult {
        uint256 numTrainers;
        uint256 evaluationsRequired;
        uint256 totalGasUsed;
        uint256 avgGasPerTrainer;
        uint256 deploymentGas;
        uint256 roundCompletionTime;
        uint256 shapleyCalculationGas;
        bool success;
        string errorMessage;
        uint256 totalTokensDistributed;
    }

    TestResult[] public testResults;

    function run() external {
        console.log("=== Federated Learning Scalability Test (FIXED) ===");
        console.log(
            "Testing trainer scalability from %s to %s trainers",
            DemoParams.MIN_TRAINERS,
            DemoParams.MAX_TRAINERS
        );
        console.log("Step size: %s", DemoParams.TRAINER_STEP_SIZE);
        console.log(
            "Sampling strategy: %s",
            DemoParams.USE_ADAPTIVE_SAMPLING ? "Adaptive (2^n)" : "Fixed"
        );
        console.log("");

        // Run scalability tests
        _runScalabilityTests();

        // Print results summary
        _printTestResults();

        // Output CSV data for graphing
        _outputCSVData();

        console.log("=== Scalability Test Completed Successfully ===");
    }

    function _runScalabilityTests() internal {
        // Test with smaller numbers first to debug the issue
        for (
            uint256 numTrainers = DemoParams.MIN_TRAINERS;
            numTrainers <= DemoParams.MAX_TRAINERS;
            numTrainers += DemoParams.TRAINER_STEP_SIZE
        ) {
            console.log("");
            console.log(
                "=== TESTING WITH %s TRAINERS ===",
                _uintToString(numTrainers)
            );

            // Use fixed sampling for debugging
            uint256 evaluationsRequired = 2 ** numTrainers;
            console.log(
                "Evaluations required: %s",
                _uintToString(evaluationsRequired)
            );

            // Setup test addresses for this configuration
            _setupTestAddressesForCount(numTrainers);

            // Deploy contracts and measure deployment gas
            uint256 deploymentGasStart = gasleft();
            bool deploymentSuccess = _deployContracts(evaluationsRequired);
            uint256 deploymentGas = deploymentGasStart - gasleft();

            TestResult memory result;
            if (deploymentSuccess) {
                // Run training demo and collect metrics
                result = _runTrainingDemoWithMetrics(
                    numTrainers,
                    evaluationsRequired,
                    deploymentGas
                );
            } else {
                result = TestResult({
                    numTrainers: numTrainers,
                    evaluationsRequired: evaluationsRequired,
                    totalGasUsed: 0,
                    avgGasPerTrainer: 0,
                    deploymentGas: deploymentGas,
                    roundCompletionTime: 0,
                    shapleyCalculationGas: 0,
                    success: false,
                    errorMessage: "Deployment failed",
                    totalTokensDistributed: 0
                });
            }

            // Store result
            testResults.push(result);

            // Print immediate results
            console.log("Test completed:");
            console.log("  Success=%s", result.success ? "YES" : "NO");
            console.log("  Total Gas=%s", _uintToString(result.totalGasUsed));
            console.log(
                "  Avg Gas/Trainer=%s",
                _uintToString(result.avgGasPerTrainer)
            );
            console.log(
                "  Tokens=%s",
                _uintToString(result.totalTokensDistributed)
            );

            if (!result.success) {
                console.log("Error: %s", result.errorMessage);
            }
        }
    }

    function _setupTestAddressesForCount(uint256 count) internal {
        // Clear existing arrays
        delete trainers;
        delete evaluators;

        // Setup aggregator
        aggregator = vm.addr(DemoParams.AGGREGATOR_KEY);

        // Setup trainers based on count
        for (uint256 i = 0; i < count; i++) {
            trainers.push(vm.addr(DemoParams.TRAINER_START_KEY + i));
        }

        // Setup evaluators (keep consistent number)
        for (uint256 i = 0; i < DemoParams.NUM_EVALUATORS; i++) {
            evaluators.push(vm.addr(DemoParams.EVALUATOR_START_KEY + i));
        }
    }

    function _deployContracts(
        uint256 evaluationsRequired
    ) internal returns (bool) {
        try this._deployContractsInternal(evaluationsRequired) {
            return true;
        } catch Error(string memory reason) {
            console.log("Deployment failed: %s", reason);
            return false;
        } catch {
            console.log("Deployment failed with unknown error");
            return false;
        }
    }

    function _deployContractsInternal(uint256 evaluationsRequired) external {
        SelectorFactory selectorFactory = new SelectorFactory(address(this));
        CalculatorFactory calculatorFactory = new CalculatorFactory(
            address(this)
        );
        AccessControlFactory accessControlFactory = new AccessControlFactory(
            address(this)
        );
        CompensationFactory compensationFactory = new CompensationFactory(
            address(this)
        );

        // Deploy implementations
        AlwaysSampled alwaysSampled = new AlwaysSampled();
        RandomSampling randomSampling = new RandomSampling();
        ContributionCalculator contributionCalculatorImpl = new ContributionCalculator();
        BaseAccessControl baseAccessControlImpl = new BaseAccessControl();
        SimpleMintCompensation simpleMintCompensationImpl = new SimpleMintCompensation();
        SwarmV1 swarmV1Impl = new SwarmV1();

        // Register implementations with factories
        selectorFactory.registerSelectorImplementation(address(alwaysSampled));
        selectorFactory.registerSelectorImplementation(address(randomSampling));
        calculatorFactory.registerCalculatorImplementation(
            address(contributionCalculatorImpl)
        );
        accessControlFactory.registerAccessControlImplementation(
            address(baseAccessControlImpl)
        );
        compensationFactory.registerCompensationImplementation(
            address(simpleMintCompensationImpl)
        );

        SwarmV1Factory swarmV1Factory = new SwarmV1Factory(
            address(swarmV1Impl),
            address(selectorFactory),
            address(calculatorFactory),
            address(accessControlFactory),
            address(compensationFactory)
        );

        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({
                name: "ScalabilityTestSwarmFixed"
            }),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: selectorFactory.getID("always-sampled-v1.0.0"),
                initData: abi.encodeWithSelector(
                    AlwaysSampled.initialize.selector
                )
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: selectorFactory.getID("random-sampling-v1.0.0"),
                initData: abi.encodeWithSelector(
                    RandomSampling.initialize.selector,
                    1 ether
                )
            }),
            contributionCalculator: SwarmV1Factory.CalculatorParams({
                id: calculatorFactory.getID("contribution-calculator-v1.0.0"),
                initData: abi.encodeWithSelector(
                    ContributionCalculator.initialize.selector,
                    address(this),
                    evaluationsRequired < 2 ? 2 : evaluationsRequired
                )
            }),
            accessControl: SwarmV1Factory.AccessControlParams({
                id: accessControlFactory.getID("base-access-control-v1.0.0"),
                initData: abi.encodeWithSelector(
                    BaseAccessControl.initialize.selector,
                    aggregator,
                    trainers,
                    evaluators
                )
            }),
            compensation: SwarmV1Factory.CompensationParams({
                id: compensationFactory.getID(
                    "simple-mint-compensation-v1.0.0"
                ),
                initData: abi.encodeWithSelector(
                    SimpleMintCompensation.initialize.selector,
                    "ScalabilityTestTokenFixed",
                    "STTF",
                    1000 ether,
                    aggregator,
                    address(this)
                )
            }),
            trainingPhaseConfiguration: BaseTrainingPhases
                .TrainingPhaseConfiguration({ttl: DemoParams.TRAINING_TTL}),
            evaluationPhaseConfiguration: BaseTrainingPhases
                .EvaluationPhaseConfiguration({
                    ttl: DemoParams.EVALUATION_TTL,
                    registrationTtl: DemoParams.EVALUATION_REGISTRATION_TTL
                })
        });

        address swarmAddress = swarmV1Factory.createSwarm(
            keccak256(
                abi.encodePacked(
                    "scalability-test-swarm-fixed",
                    block.timestamp
                )
            ),
            params
        );

        testSwarm = SwarmV1(swarmAddress);
        accessControl = BaseAccessControl(testSwarm.getAccessControl());
        compensation = SimpleMintCompensation(testSwarm.getCompensation());
        contributionCalculator = ContributionCalculator(
            testSwarm.getContributionCalculator()
        );

        // Grant necessary roles
        contributionCalculator.grantRole(
            contributionCalculator.DEFAULT_ADMIN_ROLE(),
            address(testSwarm)
        );

        contributionCalculator.grantRole(
            contributionCalculator.DEFAULT_ADMIN_ROLE(),
            aggregator
        );

        vm.startPrank(aggregator);
        compensation.grantRole(
            compensation.DEFAULT_ADMIN_ROLE(),
            address(this)
        );
        vm.stopPrank();

        compensation.grantRole(compensation.MINTER_ROLE(), address(testSwarm));
    }

    function _runTrainingDemoWithMetrics(
        uint256 numTrainers,
        uint256 evaluationsRequired,
        uint256 deploymentGas
    ) internal returns (TestResult memory) {
        uint256 roundStartTime = block.timestamp;
        uint256 totalGasUsed = 0;
        uint256 shapleyCalculationGas = 0;
        uint256 totalTokensDistributed = 0;
        bool success = false;
        string memory errorMessage = "";

        try this._runCompleteTrainingDemoWithGasTracking() returns (
            uint256 gasUsed,
            uint256 shapleyGas,
            uint256 tokensDistributed
        ) {
            totalGasUsed = gasUsed;
            shapleyCalculationGas = shapleyGas;
            totalTokensDistributed = tokensDistributed;
            success = true;
        } catch Error(string memory reason) {
            errorMessage = reason;
            console.log(
                "ERROR: Training demo failed for %s trainers: %s",
                _uintToString(numTrainers),
                reason
            );
        } catch {
            errorMessage = "Unknown error";
            console.log(
                "ERROR: Training demo failed for %s trainers with unknown error",
                _uintToString(numTrainers)
            );
        }

        uint256 roundCompletionTime = block.timestamp - roundStartTime;
        uint256 avgGasPerTrainer = numTrainers > 0
            ? totalGasUsed / numTrainers
            : 0;

        return
            TestResult({
                numTrainers: numTrainers,
                evaluationsRequired: evaluationsRequired,
                totalGasUsed: totalGasUsed,
                avgGasPerTrainer: avgGasPerTrainer,
                deploymentGas: deploymentGas,
                roundCompletionTime: roundCompletionTime,
                shapleyCalculationGas: shapleyCalculationGas,
                success: success,
                errorMessage: errorMessage,
                totalTokensDistributed: totalTokensDistributed
            });
    }

    function _runCompleteTrainingDemoWithGasTracking()
        external
        returns (
            uint256 totalGasUsed,
            uint256 shapleyGas,
            uint256 totalTokensDistributed
        )
    {
        totalGasUsed = 0;
        shapleyGas = 0;
        totalTokensDistributed = 0;
        uint256 gasUsed;
        uint256 tokensDistributed;

        console.log("=== Running Training Demo with Gas Tracking ===");

        // Run training round with gas tracking
        (
            gasUsed,
            shapleyGas,
            tokensDistributed
        ) = _runTrainingRoundWithGasTracking(1, "Scalability Test Round");
        totalGasUsed += gasUsed;
        totalTokensDistributed += tokensDistributed;

        return (totalGasUsed, shapleyGas, totalTokensDistributed);
    }

    function _runTrainingRoundWithGasTracking(
        uint256 roundId,
        string memory description
    )
        internal
        returns (
            uint256 totalGasUsed,
            uint256 shapleyGas,
            uint256 totalTokensDistributed
        )
    {
        uint256 gasUsed;
        totalGasUsed = 0;
        shapleyGas = 0;
        totalTokensDistributed = 0;

        console.log(
            "=== ROUND %s: %s ===",
            _uintToString(roundId),
            description
        );

        // Step 1: Start Training Round
        console.log("Step 1: Starting training round...");
        vm.prank(aggregator);
        gasUsed = gasleft();
        testSwarm.startTrainingRound();
        gasUsed = gasUsed - gasleft();
        totalGasUsed += gasUsed;
        console.log(" Gas used: %s", _uintToString(gasUsed));

        // Step 2: Register Trainers
        console.log("Step 2: Registering trainers...");
        gasUsed = _registerTrainersForRoundWithGas(roundId);
        totalGasUsed += gasUsed;

        // Step 3: Transition to Evaluation Phase
        console.log("Step 3: Transitioning to evaluation phase...");
        gasUsed = _transitionToEvaluationPhaseWithGas();
        totalGasUsed += gasUsed;

        // Step 4: Register Evaluators
        console.log("Step 4: Registering evaluators...");
        gasUsed = _registerEvaluatorsForRoundWithGas(roundId);
        totalGasUsed += gasUsed;

        // Step 5: Transition to Evaluation
        console.log("Step 5: Starting evaluation phase...");
        gasUsed = _transitionToEvaluationWithGas();
        totalGasUsed += gasUsed;

        // Step 6: Register Evaluations with Shapley gas tracking
        console.log("Step 6: Registering evaluations...");
        (gasUsed, shapleyGas) = _registerEvaluationsForRoundWithGas(roundId);
        totalGasUsed += gasUsed;

        // Step 7: Complete Round
        console.log("Step 7: Completing round...");
        gasUsed = _completeRoundWithGas();
        totalGasUsed += gasUsed;

        // Step 8: Claim Rewards with proper debugging
        console.log("Step 8: Claiming rewards...");
        (gasUsed, totalTokensDistributed) = _claimRewardsWithGasAndDebug(
            roundId
        );
        totalGasUsed += gasUsed;

        console.log("Round %s completed!", _uintToString(roundId));
        console.log("  Total gas: %s", _uintToString(totalGasUsed));
        console.log("  Shapley gas: %s", _uintToString(shapleyGas));
        console.log(
            "  Tokens distributed: %s",
            _uintToString(totalTokensDistributed)
        );
        return (totalGasUsed, shapleyGas, totalTokensDistributed);
    }

    // Gas tracking versions of all helper functions
    function _registerTrainersForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas) {
        totalGas = 0;
        uint256 gasUsed;

        for (uint i = 0; i < trainers.length; i++) {
            vm.prank(trainers[i]);
            gasUsed = gasleft();
            testSwarm.registerRoundContribution(
                roundId,
                keccak256(abi.encodePacked("model", i + 1, roundId))
            );
            gasUsed = gasUsed - gasleft();
            totalGas += gasUsed;
            console.log(
                " Trainer %s registered. Gas used: %s",
                _uintToString(i + 1),
                _uintToString(gasUsed)
            );
        }

        console.log(
            " Total registered trainers: %s",
            _uintToString(testSwarm.getTrainerCount(roundId))
        );
        console.log(
            " Total trainer registration gas: %s",
            _uintToString(totalGas)
        );
    }

    function _transitionToEvaluationPhaseWithGas()
        internal
        returns (uint256 gasUsed)
    {
        BaseTrainingPhases.TrainingPhaseConfiguration memory config = testSwarm
            .getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + config.ttl + 1);

        vm.prank(aggregator);
        gasUsed = gasleft();
        testSwarm.updatePhase();
        gasUsed = gasUsed - gasleft();

        console.log(" Gas used: %s", _uintToString(gasUsed));
        console.log(" Phase: %s", _getPhaseName(testSwarm.getCurrentPhase()));
    }

    function _registerEvaluatorsForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas) {
        totalGas = 0;
        uint256 gasUsed;

        for (uint i = 0; i < evaluators.length; i++) {
            vm.prank(evaluators[i]);
            gasUsed = gasleft();
            testSwarm.registerForRoundEvaluation(roundId);
            gasUsed = gasUsed - gasleft();
            totalGas += gasUsed;
            console.log(
                " Evaluator %s registered. Gas used: %s",
                _uintToString(i + 1),
                _uintToString(gasUsed)
            );
        }

        console.log(
            " Total registered evaluators: %s",
            _uintToString(testSwarm.getEvaluatorCount(roundId))
        );
        console.log(
            " Total evaluator registration gas: %s",
            _uintToString(totalGas)
        );
    }

    function _transitionToEvaluationWithGas()
        internal
        returns (uint256 gasUsed)
    {
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory config = testSwarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + config.registrationTtl + 1);

        vm.prank(aggregator);
        gasUsed = gasleft();
        testSwarm.updatePhase();
        gasUsed = gasUsed - gasleft();

        console.log(" Gas used: %s", _uintToString(gasUsed));
        console.log(" Phase: %s", _getPhaseName(testSwarm.getCurrentPhase()));
    }

    function _registerEvaluationsForRoundWithGas(
        uint256 roundId
    ) internal returns (uint256 totalGas, uint256 shapleyGas) {
        totalGas = 0;
        shapleyGas = 0;
        uint256 gasUsed;

        uint256 numTrainers = testSwarm.getTrainerCount(roundId);
        uint256 evaluationsRequired = 2 ** numTrainers; // 2^n where n = number of trainers

        // Set required evaluations for this round
        vm.prank(aggregator);
        gasUsed = gasleft();
        contributionCalculator.setEvaluationsRequired(
            roundId,
            evaluationsRequired
        );
        gasUsed = gasUsed - gasleft();
        totalGas += gasUsed;
        shapleyGas += gasUsed;

        console.log(
            " Set evaluations required: %s (gas: %s)",
            _uintToString(evaluationsRequired),
            _uintToString(gasUsed)
        );

        for (uint i = 0; i < evaluationsRequired; i++) {
            // Use round-robin assignment of evaluators
            uint256 evaluatorIndex = i % evaluators.length;
            uint256 evalId = testSwarm.getEvaluatorId(
                roundId,
                evaluators[evaluatorIndex]
            );
            uint256 taskId = testSwarm.nthTaskOfNode(roundId, evalId - 1, 0);
            uint256 mask = contributionCalculator.getMask(
                roundId,
                taskId,
                uint8(numTrainers)
            );

            vm.prank(evaluators[evaluatorIndex]);
            gasUsed = gasleft();
            testSwarm.registerEvaluation(
                roundId,
                taskId,
                mask,
                keccak256(abi.encodePacked("evaluation", i + 1, roundId)),
                int256(
                    uint256(
                        keccak256(
                            abi.encodePacked(
                                "score",
                                i,
                                roundId,
                                block.timestamp
                            )
                        )
                    ) % 101
                )
            );
            gasUsed = gasUsed - gasleft();
            totalGas += gasUsed;
            shapleyGas += gasUsed;
        }

        console.log(
            " Total evaluation registration gas: %s",
            _uintToString(totalGas)
        );
        console.log(" Shapley calculation gas: %s", _uintToString(shapleyGas));
    }

    function _completeRoundWithGas() internal returns (uint256 gasUsed) {
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory config = testSwarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + config.ttl + 1);

        vm.prank(aggregator);
        gasUsed = gasleft();
        testSwarm.updatePhase();
        gasUsed = gasUsed - gasleft();

        console.log(" Gas used: %s", _uintToString(gasUsed));
    }

    function _claimRewardsWithGasAndDebug(
        uint256 roundId
    ) internal returns (uint256 totalGas, uint256 totalTokensDistributed) {
        totalGas = 0;
        totalTokensDistributed = 0;
        uint256 gasUsed;

        uint256 numTrainers = testSwarm.getTrainerCount(roundId);

        for (uint i = 0; i < numTrainers; i++) {
            uint256 balanceBefore = compensation.balanceOf(trainers[i]);

            // Debug: Check Shapley value before claiming
            int256 shapleyValue = contributionCalculator.calculateContribution(
                roundId,
                i, // trainer index (0-based)
                uint8(numTrainers)
            );
            console.log(
                " Trainer %s Shapley value: %s",
                _uintToString(i + 1),
                _intToString(shapleyValue)
            );

            vm.prank(trainers[i]);
            gasUsed = gasleft();
            testSwarm.claimReward(uint64(roundId), trainers[i]);
            gasUsed = gasUsed - gasleft();
            totalGas += gasUsed;

            uint256 balanceAfter = compensation.balanceOf(trainers[i]);
            uint256 rewardReceived = balanceAfter > balanceBefore
                ? balanceAfter - balanceBefore
                : 0;
            totalTokensDistributed += rewardReceived;

            console.log(
                " Trainer %s claimed reward: %s tokens. Gas used: %s",
                _uintToString(i + 1),
                _uintToString(rewardReceived),
                _uintToString(gasUsed)
            );
        }

        uint256 avgGasPerTrainer = totalGas / numTrainers;
        console.log(" Total reward claiming gas: %s", _uintToString(totalGas));
        console.log(
            " Average gas per trainer: %s",
            _uintToString(avgGasPerTrainer)
        );
        console.log(
            " Total tokens distributed: %s",
            _uintToString(totalTokensDistributed)
        );
    }

    function _printTestResults() internal view {
        console.log("");
        console.log("=== SCALABILITY TEST RESULTS SUMMARY ===");
        console.log("Total tests run: %s", _uintToString(testResults.length));

        uint256 successfulTests = 0;
        uint256 totalGasUsed = 0;
        uint256 maxGasUsed = 0;
        uint256 minGasUsed = type(uint256).max;
        uint256 totalTokensDistributed = 0;

        for (uint i = 0; i < testResults.length; i++) {
            TestResult memory result = testResults[i];

            if (result.success) {
                successfulTests++;
                totalGasUsed += result.totalGasUsed;
                totalTokensDistributed += result.totalTokensDistributed;
                if (result.totalGasUsed > maxGasUsed)
                    maxGasUsed = result.totalGasUsed;
                if (result.totalGasUsed < minGasUsed)
                    minGasUsed = result.totalGasUsed;
            }

            console.log("Trainers: %s", _uintToString(result.numTrainers));
            console.log("  Success: %s", result.success ? "YES" : "NO");
            console.log("  Gas: %s", _uintToString(result.totalGasUsed));
            console.log(
                "  Avg Gas/Trainer: %s",
                _uintToString(result.avgGasPerTrainer)
            );
            console.log(
                "  Tokens: %s",
                _uintToString(result.totalTokensDistributed)
            );
        }

        console.log("");
        console.log(
            "Success rate: %s%%",
            _uintToString((successfulTests * 100) / testResults.length)
        );
        console.log("Total gas used: %s", _uintToString(totalGasUsed));
        console.log(
            "Total tokens distributed: %s",
            _uintToString(totalTokensDistributed)
        );
        console.log("Max gas used: %s", _uintToString(maxGasUsed));
        console.log("Min gas used: %s", _uintToString(minGasUsed));
    }

    function _outputCSVData() internal view {
        console.log("");
        console.log("=== CSV DATA FOR GRAPHING (TABLE) ===");
        string
            memory header = "| numTrainers | evaluationsRequired | totalGasUsed | avgGasPerTrainer | deploymentGas | roundCompletionTime | shapleyCalculationGas | totalTokensDistributed | success |";
        string
            memory separator = "|-------------|--------------------|--------------|------------------|---------------|---------------------|-----------------------|------------------------|---------|";
        console.log("%s", header);
        console.log("%s", separator);

        for (uint i = 0; i < testResults.length; i++) {
            TestResult memory result = testResults[i];
            // Print row in table format (all values as strings)
            // Workaround to avoid too many arguments in console.log
            string memory row = string(
                abi.encodePacked(
                    "| ",
                    _uintToString(result.numTrainers),
                    " | ",
                    _uintToString(result.evaluationsRequired),
                    " | ",
                    _uintToString(result.totalGasUsed),
                    " | ",
                    _uintToString(result.avgGasPerTrainer),
                    " | ",
                    _uintToString(result.deploymentGas),
                    " | ",
                    _uintToString(result.roundCompletionTime),
                    " | ",
                    _uintToString(result.shapleyCalculationGas),
                    " | ",
                    _uintToString(result.totalTokensDistributed),
                    " | ",
                    result.success ? "true" : "false",
                    " |"
                )
            );
            console.log("%s", row);
        }
    }

    function _getPhaseName(
        bytes32 phase
    ) internal pure returns (string memory) {
        if (phase == keccak256("IDLE")) return "IDLE";
        if (phase == keccak256("TRAINING")) return "TRAINING";
        if (phase == keccak256("EVALUATION")) return "EVALUATION";
        return "UNKNOWN";
    }

    function _uintToString(
        uint256 value
    ) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }

        uint256 temp = value;
        uint256 digits;

        while (temp != 0) {
            digits++;
            temp /= 10;
        }

        bytes memory buffer = new bytes(digits);

        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }

        return string(buffer);
    }

    function _intToString(int256 value) internal pure returns (string memory) {
        if (value == 0) {
            return "0";
        }

        bool negative = value < 0;
        if (negative) {
            value = -value;
        }

        string memory result = _uintToString(uint256(value));

        if (negative) {
            result = string(abi.encodePacked("-", result));
        }

        return result;
    }
}
