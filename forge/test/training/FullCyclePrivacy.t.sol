// SPDX-License-Identifier: UNLICENSED

pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "forge-std/console.sol";

import {SwarmV2} from "@rizemind-contracts/swarm/SwarmV2.sol";
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

/// @title FullCyclePrivacyTest
/// @notice Exercises privacy-preserving trainer commitments with slashing
contract FullCyclePrivacy is Test {
    struct PrivacyRecord {
        bytes32 commitment;
        bytes32 nonce;
        uint64 deadline;
    }

    SwarmV2 public swarm;
    BaseAccessControl public accessControl;
    SimpleMintCompensation public compensation;
    ContributionCalculator public calculator;

    address public aggregator;
    address[] public trainers;
    address[] public evaluators;

    mapping(address => PrivacyRecord) private privacyRecords;

    mapping(address => uint256) private trainerRewards;

    uint256 private aggregatorBondDeposit;
    uint256 private slashPenalty;
    uint256 private slashFinderReward;

    address private pendingRevealTrainer;

    uint256 private constant TRAINER_COUNT = 3;
    uint256 private constant EVALUATOR_COUNT = 2;
    uint64 private constant REVEAL_BUFFER = 15 minutes;
    uint256 private constant PRIVACY_PENALTY = 0.5 ether;
    uint16 private constant FINDER_REWARD_BPS = 1_000; // 10%

    function setUp() public {
        console.log("=== Swarm Privacy Mode Test Setup ===");

        slashPenalty = 0;
        slashFinderReward = 0;
        aggregatorBondDeposit = 0;
        pendingRevealTrainer = address(0);

        _setupActors();
        _deployStack();
        _configurePrivacy();
    }

    function testFullCyclePrivacyRound() public {
        console.log("=== Swarm Privacy Mode Demo ===");

        _executeRound();
        _assertRoundOutcome();
        _printSummary();

        console.log("=== Demo Completed ===");
    }

    /// @notice Prepare aggregator, trainer, and evaluator addresses
    function _setupActors() internal {
        delete trainers;
        delete evaluators;

        aggregator = vm.addr(DemoParams.AGGREGATOR_KEY);

        for (uint256 i = 0; i < TRAINER_COUNT; ++i) {
            trainers.push(vm.addr(DemoParams.TRAINER_START_KEY + i));
            trainerRewards[trainers[i]] = 0;
        }

        for (uint256 i = 0; i < EVALUATOR_COUNT; ++i) {
            evaluators.push(vm.addr(DemoParams.EVALUATOR_START_KEY + i));
        }
    }

    /// @notice Deploy a fresh Swarm stack for the demo
    function _deployStack() internal {
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

        AlwaysSampled alwaysSampled = new AlwaysSampled();
        RandomSampling randomSampling = new RandomSampling();
        ContributionCalculator calculatorImpl = new ContributionCalculator();
        BaseAccessControl accessImpl = new BaseAccessControl();
        SimpleMintCompensation compensationImpl = new SimpleMintCompensation();
        SwarmV2 swarmImpl = new SwarmV2();

        selectorFactory.registerSelectorImplementation(address(alwaysSampled));
        selectorFactory.registerSelectorImplementation(address(randomSampling));
        calculatorFactory.registerCalculatorImplementation(
            address(calculatorImpl)
        );
        accessControlFactory.registerAccessControlImplementation(
            address(accessImpl)
        );
        compensationFactory.registerCompensationImplementation(
            address(compensationImpl)
        );

        SwarmV1Factory factory = new SwarmV1Factory(
            address(swarmImpl),
            address(selectorFactory),
            address(calculatorFactory),
            address(accessControlFactory),
            address(compensationFactory)
        );

        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({name: "privacy-demo"}),
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
                    DemoParams.RANDOM_SAMPLING_INITIAL_STAKE
                )
            }),
            contributionCalculator: SwarmV1Factory.CalculatorParams({
                id: calculatorFactory.getID("contribution-calculator-v1.0.0"),
                initData: abi.encodeWithSelector(
                    ContributionCalculator.initialize.selector,
                    address(this),
                    uint256(2 ** TRAINER_COUNT)
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
                    "PrivacyToken",
                    "PRIV",
                    1_000 ether,
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

        address swarmAddress = factory.createSwarm(
            keccak256(abi.encodePacked("privacy-demo", block.timestamp)),
            params
        );

        swarm = SwarmV2(swarmAddress);
        accessControl = BaseAccessControl(swarm.getAccessControl());
        compensation = SimpleMintCompensation(swarm.getCompensation());
        calculator = ContributionCalculator(swarm.getContributionCalculator());

        calculator.grantRole(calculator.DEFAULT_ADMIN_ROLE(), address(swarm));
        calculator.grantRole(calculator.DEFAULT_ADMIN_ROLE(), aggregator);

        vm.startPrank(aggregator);
        compensation.grantRole(
            compensation.DEFAULT_ADMIN_ROLE(),
            address(this)
        );
        compensation.grantRole(compensation.MINTER_ROLE(), address(swarm));
        vm.stopPrank();
    }

    /// @notice Configure privacy settings and fund the aggregator bond
    function _configurePrivacy() internal {
        vm.prank(aggregator);
        swarm.configureTrainerPrivacy(PRIVACY_PENALTY, FINDER_REWARD_BPS);

        vm.prank(aggregator);
        swarm.setTrainerPrivacyMode(true);

        uint256 depositAmount = PRIVACY_PENALTY * TRAINER_COUNT;
        aggregatorBondDeposit = depositAmount;
        if (depositAmount > 0) {
            vm.deal(aggregator, depositAmount);
            vm.prank(aggregator);
            swarm.depositAggregatorBond{value: depositAmount}();
        }

        (uint256 balance, uint256 reserved) = swarm.getAggregatorBondState();
        console.log(
            "Aggregator bond funded: balance=%s reserved=%s",
            _uintToString(balance),
            _uintToString(reserved)
        );
    }

    /// @notice Execute a single training round demonstrating privacy mode
    function _executeRound() internal {
        console.log("\n=== Executing Round 1 ===");

        vm.prank(aggregator);
        swarm.startTrainingRound();

        _commitTrainers(1);
        _advanceThroughTrainingTTL();
        _registerEvaluators(1);
        _advanceThroughEvaluationRegistration();
        _registerEvaluations(1);
        _revealCommittedTrainers(1);
        _handleMissedReveal(1);
        _advanceThroughEvaluationTTL();
        _claimRewards(1);
    }

    function _commitTrainers(uint256 roundId) internal {
        console.log("Step 1: Aggregator commits trainers with privacy...");
        for (uint256 i = 0; i < trainers.length; ++i) {
            address trainer = trainers[i];
            bytes32 nonce = keccak256(
                abi.encodePacked("privacy", trainer, roundId, i, block.number)
            );
            uint64 deadline = uint64(
                block.timestamp +
                    DemoParams.TRAINING_TTL +
                    DemoParams.EVALUATION_REGISTRATION_TTL +
                    DemoParams.EVALUATION_TTL +
                    REVEAL_BUFFER
            );
            bytes32 commitment = keccak256(abi.encodePacked(trainer, nonce));

            privacyRecords[trainer] = PrivacyRecord({
                commitment: commitment,
                nonce: nonce,
                deadline: deadline
            });

            bytes32 modelHash = keccak256(
                abi.encodePacked("model", i + 1, roundId)
            );
            vm.prank(aggregator);
            swarm.registerRoundContributionPrivacy(
                roundId,
                commitment,
                modelHash,
                deadline
            );
            console.log(
                " Committed trainer %s with deadline %s",
                _uintToString(i + 1),
                _uintToString(deadline)
            );
        }

        (uint256 balance, uint256 reserved) = swarm.getAggregatorBondState();
        console.log(
            " Bond state after commitments: balance=%s reserved=%s",
            _uintToString(balance),
            _uintToString(reserved)
        );
    }

    function _advanceThroughTrainingTTL() internal {
        BaseTrainingPhases.TrainingPhaseConfiguration memory config = swarm
            .getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + config.ttl + 1);
        vm.prank(aggregator);
        swarm.updatePhase();
        console.log(
            "Step 2: Training phase closed, moved to evaluator registration."
        );
    }

    function _registerEvaluators(uint256 roundId) internal {
        console.log("Step 3: Evaluators register...");
        for (uint256 i = 0; i < evaluators.length; ++i) {
            vm.prank(evaluators[i]);
            swarm.registerForRoundEvaluation(roundId);
            console.log(" Evaluator %s registered", _uintToString(i + 1));
        }
    }

    function _advanceThroughEvaluationRegistration() internal {
        BaseTrainingPhases.EvaluationPhaseConfiguration memory config = swarm
            .getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + config.registrationTtl + 1);
        vm.prank(aggregator);
        swarm.updatePhase();
        console.log("Step 4: Evaluation phase started.");
    }

    function _registerEvaluations(uint256 roundId) internal {
        console.log("Step 5: Evaluators submit results...");

        uint256 numTrainers = trainers.length;
        uint256 evaluationsRequired = 2 ** numTrainers;

        vm.prank(aggregator);
        calculator.setEvaluationsRequired(roundId, evaluationsRequired);

        for (uint256 i = 0; i < evaluationsRequired; ++i) {
            address evaluator = evaluators[i % evaluators.length];
            uint256 evaluatorId = swarm.getEvaluatorId(roundId, evaluator);
            uint256 taskId = swarm.nthTaskOfNode(roundId, evaluatorId - 1, 0);
            uint256 mask = calculator.getMask(
                roundId,
                taskId,
                uint8(numTrainers)
            );

            vm.prank(evaluator);
            swarm.registerEvaluation(
                roundId,
                taskId,
                mask,
                keccak256(abi.encodePacked("evaluation", i + 1, roundId)),
                int256(
                    uint256(keccak256(abi.encodePacked("score", i, roundId))) %
                        100
                )
            );
        }
    }

    function _revealCommittedTrainers(uint256 roundId) internal {
        console.log("Step 6: Aggregator reveals trainer identities...");
        pendingRevealTrainer = trainers[trainers.length - 1];

        for (uint256 i = 0; i < trainers.length - 1; ++i) {
            address trainer = trainers[i];
            PrivacyRecord memory record = privacyRecords[trainer];
            vm.prank(aggregator);
            swarm.revealTrainerCommitment(
                roundId,
                trainer,
                abi.encodePacked(record.nonce)
            );
            console.log(" Trainer %s revealed", _uintToString(i + 1));
        }

        console.log(
            " Deliberately keeping trainer %s hidden to trigger slashing...",
            _uintToString(trainers.length)
        );
    }

    function _handleMissedReveal(uint256 roundId) internal {
        if (pendingRevealTrainer == address(0)) {
            return;
        }

        PrivacyRecord memory record = privacyRecords[pendingRevealTrainer];
        vm.warp(record.deadline + 1);

        address finder = evaluators[0];
        vm.prank(finder);
        (uint256 penalty, uint256 reward) = swarm.slashTrainerCommitment(
            roundId,
            record.commitment
        );
        slashPenalty = penalty;
        slashFinderReward = reward;
        console.log(
            "Step 7: Finder %s slashed missed reveal. Penalty=%s Reward=%s",
            _addressToString(finder),
            _uintToString(penalty),
            _uintToString(reward)
        );

        vm.prank(aggregator);
        swarm.revealTrainerCommitment(
            roundId,
            pendingRevealTrainer,
            abi.encodePacked(record.nonce)
        );
        console.log(" Aggregator revealed final trainer after slashing.");

        pendingRevealTrainer = address(0);
    }

    function _advanceThroughEvaluationTTL() internal {
        BaseTrainingPhases.EvaluationPhaseConfiguration memory config = swarm
            .getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + config.ttl + 1);
        vm.prank(aggregator);
        swarm.updatePhase();
        console.log("Step 8: Round finalized.");
    }

    function _claimRewards(uint256 roundId) internal {
        console.log("Step 9: Trainers claim rewards...");
        for (uint256 i = 0; i < trainers.length; ++i) {
            address trainer = trainers[i];
            uint256 beforeBalance = compensation.balanceOf(trainer);
            vm.prank(trainer);
            swarm.claimReward(roundId, trainer);
            uint256 received = compensation.balanceOf(trainer) - beforeBalance;
            trainerRewards[trainer] = received;
            console.log(
                " Trainer %s received %s tokens",
                _uintToString(i + 1),
                _uintToString(received)
            );
        }
    }

    function _assertRoundOutcome() internal view {
        (uint256 balance, uint256 reserved) = swarm.getAggregatorBondState();
        assertEq(reserved, 0, "Aggregator bond should have no reservations");
        assertEq(slashPenalty, PRIVACY_PENALTY, "Unexpected slash penalty");
        assertEq(
            slashFinderReward,
            (slashPenalty * FINDER_REWARD_BPS) / 10_000,
            "Finder reward mismatch"
        );
        assertEq(
            balance + slashPenalty,
            aggregatorBondDeposit,
            "Aggregator bond balance mismatch"
        );

        for (uint256 i = 0; i < trainers.length; ++i) {
            assertGt(trainerRewards[trainers[i]], 0, "Trainer reward missing");
        }

        assertEq(
            pendingRevealTrainer,
            address(0),
            "Pending reveal should be cleared"
        );
    }

    function _printSummary() internal view {
        (uint256 balance, uint256 reserved) = swarm.getAggregatorBondState();
        console.log(
            "\nFinal aggregator bond: balance=%s reserved=%s",
            _uintToString(balance),
            _uintToString(reserved)
        );
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

    function _addressToString(
        address account
    ) internal pure returns (string memory) {
        bytes16 hexSymbols = 0x30313233343536373839616263646566;
        bytes20 addrBytes = bytes20(account);
        bytes memory buffer = new bytes(42);
        buffer[0] = "0";
        buffer[1] = "x";
        for (uint256 i = 0; i < 20; ++i) {
            uint8 byteValue = uint8(addrBytes[i]);
            buffer[2 + (i << 1)] = bytes1(hexSymbols[byteValue >> 4]);
            buffer[3 + (i << 1)] = bytes1(hexSymbols[byteValue & 0x0f]);
        }
        return string(buffer);
    }
}
