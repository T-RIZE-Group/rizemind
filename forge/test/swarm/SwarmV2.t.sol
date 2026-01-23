// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {SwarmV2} from "../../src/swarm/SwarmV2.sol";
import {SwarmV2Factory} from "../../src/swarm/SwarmV2Factory.sol";
import {SelectorFactory} from "../../src/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "../../src/contribution/CalculatorFactory.sol";
import {AlwaysSampled} from "../../src/sampling/AlwaysSampled.sol";
import {RandomSampling} from "../../src/sampling/RandomSampling.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {BaseTrainingPhases} from "../../src/training/BaseTrainingPhases.sol";
import {BaseAccessControl} from "../../src/access/BaseAccessControl.sol";
import {SimpleMintCompensation} from "../../src/compensation/SimpleMintCompensation.sol";
import {AccessControlFactory} from "../../src/access/AccessControlFactory.sol";
import {CompensationFactory} from "../../src/compensation/CompensationFactory.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {RoundTrainerRegistryV2} from "../../src/swarm/registry/RoundTrainerRegistryV2.sol";

contract SwarmV2Test is Test {
    SwarmV2 public implementation;
    SwarmV2Factory public factory;
    SelectorFactory public selectorFactory;
    CalculatorFactory public calculatorFactory;
    AccessControlFactory public accessControlFactory;
    CompensationFactory public compensationFactory;
    SwarmV2 public swarm;

    // Implementation contracts
    AlwaysSampled public trainerSelectorImpl;
    RandomSampling public evaluatorSelectorImpl;
    ContributionCalculator public calculatorImpl;
    BaseAccessControl public accessControlImpl;
    SimpleMintCompensation public compensationImpl;

    address public aggregator = address(0x1);
    address public trainer1 = address(0x2);
    address public trainer2 = address(0x3);
    address public trainer3 = address(0x4);
    address public evaluator1 = address(0x5);
    address public evaluator2 = address(0x6);

    address[] public initialTrainers;
    address[] public initialEvaluators;
    // Factory IDs
    bytes32 TRAINER_SELECTOR_ID;
    bytes32 EVALUATOR_SELECTOR_ID;
    bytes32 CALCULATOR_ID;
    bytes32 ACCESS_CONTROL_ID;
    bytes32 COMPENSATION_ID;

    function setUp() public {
        // Deploy implementation
        implementation = new SwarmV2();

        // Deploy selector factory
        selectorFactory = new SelectorFactory(address(this));

        // Deploy calculator factory
        calculatorFactory = new CalculatorFactory(address(this));

        // Deploy access control factory
        accessControlFactory = new AccessControlFactory(address(this));

        // Deploy compensation factory
        compensationFactory = new CompensationFactory(address(this));

        // Deploy and register selector implementations
        trainerSelectorImpl = new AlwaysSampled();
        evaluatorSelectorImpl = new RandomSampling();

        (, , string memory version, , , , ) = trainerSelectorImpl
            .eip712Domain();
        TRAINER_SELECTOR_ID = selectorFactory.getID(version);
        (, , string memory version2, , , , ) = evaluatorSelectorImpl
            .eip712Domain();
        EVALUATOR_SELECTOR_ID = selectorFactory.getID(version2);

        selectorFactory.registerSelectorImplementation(
            address(trainerSelectorImpl)
        );
        selectorFactory.registerSelectorImplementation(
            address(evaluatorSelectorImpl)
        );

        // Deploy and register calculator implementation
        calculatorImpl = new ContributionCalculator();
        (, , string memory version3, , , , ) = calculatorImpl.eip712Domain();
        CALCULATOR_ID = calculatorFactory.getID(version3);
        calculatorFactory.registerCalculatorImplementation(
            address(calculatorImpl)
        );

        // Deploy and register access control implementation
        accessControlImpl = new BaseAccessControl();
        (, , string memory version4, , , , ) = accessControlImpl.eip712Domain();
        ACCESS_CONTROL_ID = accessControlFactory.getID(version4);
        accessControlFactory.registerAccessControlImplementation(
            address(accessControlImpl)
        );

        // Deploy and register compensation implementation
        compensationImpl = new SimpleMintCompensation();
        (, , string memory version5, , , , ) = compensationImpl.eip712Domain();
        COMPENSATION_ID = compensationFactory.getID(version5);
        compensationFactory.registerCompensationImplementation(
            address(compensationImpl)
        );

        // Deploy factory
        factory = new SwarmV2Factory(
            address(implementation),
            address(selectorFactory),
            address(calculatorFactory),
            address(accessControlFactory),
            address(compensationFactory)
        );

        // Set up initial trainers
        initialTrainers = new address[](3);
        initialTrainers[0] = trainer1;
        initialTrainers[1] = trainer2;
        initialTrainers[2] = trainer3;

        initialEvaluators = new address[](2);
        initialEvaluators[0] = evaluator1;
        initialEvaluators[1] = evaluator2;

        address swarmAddress = factory.getSwarmAddress(keccak256("test-salt"));

        //TODO: add evaluators to access control

        // Create swarm using factory
        SwarmV2Factory.SwarmParams memory params = SwarmV2Factory.SwarmParams({
            swarm: SwarmV2Factory.SwarmV2Params({name: "TestSwarm"}),
            trainerSelector: SwarmV2Factory.SelectorParams({
                id: TRAINER_SELECTOR_ID,
                initData: abi.encodeWithSelector(
                    AlwaysSampled.initialize.selector
                )
            }),
            evaluatorSelector: SwarmV2Factory.SelectorParams({
                id: EVALUATOR_SELECTOR_ID,
                initData: abi.encodeWithSelector(
                    RandomSampling.initialize.selector,
                    1 ether
                ) // 100% selection rate
            }),
            contributionCalculator: SwarmV2Factory.CalculatorParams({
                id: CALCULATOR_ID,
                initData: abi.encodeWithSelector(
                    ContributionCalculator.initialize.selector,
                    swarmAddress,
                    2
                )
            }),
            accessControl: SwarmV2Factory.AccessControlParams({
                id: ACCESS_CONTROL_ID,
                initData: abi.encodeWithSelector(
                    BaseAccessControl.initialize.selector,
                    aggregator,
                    initialTrainers,
                    initialEvaluators
                )
            }),
            compensation: SwarmV2Factory.CompensationParams({
                id: COMPENSATION_ID,
                initData: abi.encodeWithSelector(
                    SimpleMintCompensation.initialize.selector,
                    "TestToken",
                    "TST",
                    1000 ether,
                    aggregator,
                    swarmAddress
                )
            }),
            trainingPhaseConfiguration: BaseTrainingPhases
                .TrainingPhaseConfiguration({ttl: 1000}),
            evaluationPhaseConfiguration: BaseTrainingPhases
                .EvaluationPhaseConfiguration({
                    ttl: 1000,
                    registrationTtl: 1000
                })
        });

        swarm = SwarmV2(factory.createSwarm(keccak256("test-salt"), params));

        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );
    }

    // ============================================================================
    // INITIALIZATION TESTS
    // ============================================================================

    function test_initialize_wrongInitialization() public {
        // Test that calling initialize() without parameters reverts
        vm.expectRevert(SwarmV2.WrongInitialization.selector);
        swarm.initialize();
    }

    // ============================================================================
    // ACCESS CONTROL TESTS
    // ============================================================================

    function test_onlyAggregator() public {
        // Test that only aggregator can call aggregator-only functions
        vm.prank(trainer1);
        vm.expectRevert();
        swarm.startTrainingRound();

        vm.prank(aggregator);
        // Should not revert
        swarm.startTrainingRound();
    }

    function test_onlyTrainer() public {
        // Test that only trainers can call trainer-only functions
        vm.prank(evaluator1);
        vm.expectRevert();
        swarm.registerRoundContribution(1, keccak256("model1"));

        vm.prank(trainer1);
        // Should not revert (but will revert due to phase)
        vm.expectRevert(SwarmV2.NotTrainingPhase.selector);
        swarm.registerRoundContribution(1, keccak256("model1"));
    }

    function test_onlyEvaluator() public {
        // Test that only evaluators can call evaluator-only functions
        vm.prank(trainer1);
        vm.expectRevert();
        swarm.registerForRoundEvaluation(1);

        vm.prank(evaluator1);
        // Should not revert (but will revert due to phase)
        vm.expectRevert(SwarmV2.NotEvaluatorRegistrationPhase.selector);
        swarm.registerForRoundEvaluation(1);
    }

    // ============================================================================
    // TRAINING ROUND TESTS
    // ============================================================================

    function test_startTrainingRound() public {
        // Test starting a training round
        assertTrue(swarm.isIdle(), "Should start in idle phase");

        vm.prank(aggregator);
        swarm.startTrainingRound();

        assertTrue(
            swarm.isTraining(),
            "Should be in training phase after starting round"
        );
    }

    function test_startTrainingRound_notIdle() public {
        // Test starting training round when not idle
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Try to start another round
        vm.prank(aggregator);
        vm.expectRevert(SwarmV2.NotIdle.selector);
        swarm.startTrainingRound();
    }

    function test_registerRoundContribution() public {
        // Start training round
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register contribution
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));

        // Check that trainer is registered
        assertTrue(
            swarm.isTrainerRegistered(1, trainer1),
            "Trainer should be registered"
        );
        assertEq(
            swarm.getTrainerId(1, trainer1),
            1,
            "Trainer should have ID 1"
        );
    }

    function test_registerRoundContribution_notTrainingPhase() public {
        // Try to register contribution when not in training phase
        vm.prank(trainer1);
        vm.expectRevert(SwarmV2.NotTrainingPhase.selector);
        swarm.registerRoundContribution(1, keccak256("model1"));
    }

    // ============================================================================
    // EVALUATION REGISTRATION TESTS
    // ============================================================================

    function test_registerForRoundEvaluation() public {
        // Start training round and fast forward to evaluator registration phase
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register some trainers first
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        // Fast forward to evaluator registration phase
        vm.warp(block.timestamp + trainingConfig.ttl); // Past training TTL
        //swarm.updatePhase(); we intentionally don't call updatePhase here to test automated phase transition

        // Register evaluator
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);

        // Check that evaluator is registered
        assertTrue(
            swarm.isEvaluatorRegistered(1, evaluator1),
            "Evaluator should be registered"
        );
        assertEq(
            swarm.getEvaluatorId(1, evaluator1),
            1,
            "Evaluator should have ID 1"
        );
    }

    function test_registerForRoundEvaluation_notRegistrationPhase() public {
        // Try to register when not in evaluator registration phase
        vm.prank(evaluator1);
        vm.expectRevert(SwarmV2.NotEvaluatorRegistrationPhase.selector);
        swarm.registerForRoundEvaluation(1);
    }

    // ============================================================================
    // EVALUATION TESTS
    // ============================================================================

    function test_registerEvaluation() public {
        // Complete the full flow to evaluation phase
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register trainers
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();

        // Fast forward to evaluator registration phase
        vm.warp(block.timestamp + trainingConfig.ttl);

        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);

        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationPhaseConfiguration = swarm
                .getEvaluationPhaseConfiguration();
        // Fast forward to evaluation phase
        vm.warp(block.timestamp + evaluationPhaseConfiguration.registrationTtl); // Past registration TTL

        // Register evaluation
        vm.startPrank(evaluator1);
        swarm.updatePhase();
        uint256 evalId = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId = swarm.nthTaskOfNode(1, evalId - 1, 0);

        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );
        uint256 mask = contributionCalculator.getMask(1, taskId, 2);
        swarm.registerEvaluation(1, taskId, mask, keccak256("model1"), 100);
        vm.stopPrank();
        // Should not revert
        assertTrue(true, "Evaluation should be registered successfully");
    }

    function test_registerEvaluation_notEvaluationPhase() public {
        // Try to register evaluation when not in evaluation phase
        vm.prank(evaluator1);
        vm.expectRevert(SwarmV2.NotEvaluationPhase.selector);
        swarm.registerEvaluation(1, 1, 1, keccak256("model1"), 100);
    }

    function test_registerEvaluation_notAssigned() public {
        // Complete the full flow to evaluation phase
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register trainers
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));

        // Fast forward to evaluator registration phase using actual config
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();

        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);

        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);

        // Fast forward to evaluation phase using actual config
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();

        // using evaluator2's task
        uint256 evalId = swarm.getEvaluatorId(1, evaluator2);
        uint256 taskId = swarm.nthTaskOfNode(1, evalId - 1, 0);

        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );
        uint256 mask = contributionCalculator.getMask(1, taskId, 2);
        // Try to register evaluation for task not assigned to evaluator
        vm.prank(evaluator1);
        vm.expectRevert(
            abi.encodeWithSelector(
                SwarmV2.NotAssignedTo.selector,
                1,
                taskId,
                evaluator1
            )
        );
        swarm.registerEvaluation(1, taskId, mask, keccak256("model1"), 100);
    }

    // ============================================================================
    // REWARD CLAIMING TESTS
    // ============================================================================

    function test_claimReward() public {
        // Complete the full flow
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register trainers
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));

        // Fast forward to evaluator registration phase
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();

        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);

        // Fast forward to evaluation phase
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();

        // Register evaluations with proper flow
        vm.startPrank(evaluator1);
        uint256 evalId1 = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId1 = swarm.nthTaskOfNode(1, evalId1 - 1, 0);
        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );
        uint256 mask1 = contributionCalculator.getMask(1, taskId1, 2);
        swarm.registerEvaluation(1, taskId1, mask1, keccak256("model1"), 100);
        vm.stopPrank();

        vm.startPrank(evaluator2);
        uint256 evalId2 = swarm.getEvaluatorId(1, evaluator2);
        uint256 taskId2 = swarm.nthTaskOfNode(1, evalId2 - 1, 0);
        uint256 mask2 = contributionCalculator.getMask(1, taskId2, 2);
        swarm.registerEvaluation(1, taskId2, mask2, keccak256("model2"), 200);
        vm.stopPrank();

        // Fast forward to idle
        vm.warp(block.timestamp + evaluationConfig.ttl);
        swarm.updatePhase();

        // Claim reward
        swarm.claimReward(1, trainer1);

        // Should not revert
        assertTrue(true, "Reward should be claimed successfully");
    }

    function test_registerRoundContributionPrivacy_revertsWhenDisabled()
        public
    {
        vm.prank(aggregator);
        vm.expectRevert(SwarmV2.PrivacyModeDisabled.selector);
        swarm.registerRoundContributionPrivacy(
            1,
            keccak256("commit"),
            keccak256("model"),
            uint64(block.timestamp + 1 hours)
        );
    }

    function test_registerRoundContribution_revertsWhenPrivacyEnabled() public {
        vm.prank(aggregator);
        swarm.configureTrainerPrivacy(1 ether, 0);
        vm.prank(aggregator);
        swarm.setTrainerPrivacyMode(true);

        vm.prank(trainer1);
        vm.expectRevert(SwarmV2.PrivacyModeEnabled.selector);
        swarm.registerRoundContribution(1, keccak256("model1"));
    }

    function test_claimRewardRequiresReveal() public {
        uint256 penalty = 1 ether;
        vm.prank(aggregator);
        swarm.configureTrainerPrivacy(penalty, 0);
        vm.prank(aggregator);
        swarm.setTrainerPrivacyMode(true);

        vm.deal(aggregator, 5 ether);
        vm.prank(aggregator);
        swarm.depositAggregatorBond{value: 5 ether}();

        vm.prank(aggregator);
        swarm.startTrainingRound();

        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();

        bytes32 modelHash1 = keccak256("model1");
        bytes32 modelHash2 = keccak256("model2");
        bytes32 nonce1 = keccak256("nonce1");
        bytes32 nonce2 = keccak256("nonce2");
        uint64 revealDeadline = uint64(
            block.timestamp +
                trainingConfig.ttl +
                evaluationConfig.registrationTtl +
                evaluationConfig.ttl +
                1 hours
        );

        vm.prank(aggregator);
        swarm.registerRoundContributionPrivacy(
            1,
            keccak256(abi.encodePacked(trainer1, nonce1)),
            modelHash1,
            revealDeadline
        );
        vm.prank(aggregator);
        swarm.registerRoundContributionPrivacy(
            1,
            keccak256(abi.encodePacked(trainer2, nonce2)),
            modelHash2,
            revealDeadline
        );

        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();

        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);

        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();

        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );

        vm.startPrank(evaluator1);
        uint256 evalId1 = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId1 = swarm.nthTaskOfNode(1, evalId1 - 1, 0);
        uint256 mask1 = contributionCalculator.getMask(1, taskId1, 2);
        swarm.registerEvaluation(1, taskId1, mask1, modelHash1, 100);
        vm.stopPrank();

        vm.startPrank(evaluator2);
        uint256 evalId2 = swarm.getEvaluatorId(1, evaluator2);
        uint256 taskId2 = swarm.nthTaskOfNode(1, evalId2 - 1, 0);
        uint256 mask2 = contributionCalculator.getMask(1, taskId2, 2);
        swarm.registerEvaluation(1, taskId2, mask2, modelHash2, 200);
        vm.stopPrank();

        vm.warp(block.timestamp + evaluationConfig.ttl);
        swarm.updatePhase();

        vm.expectRevert(
            abi.encodeWithSelector(
                RoundTrainerRegistryV2.TrainerNotFound.selector,
                1,
                trainer1
            )
        );
        swarm.claimReward(1, trainer1);

        vm.prank(aggregator);
        swarm.revealTrainerCommitment(1, trainer1, abi.encodePacked(nonce1));
        vm.prank(aggregator);
        swarm.revealTrainerCommitment(1, trainer2, abi.encodePacked(nonce2));

        swarm.claimReward(1, trainer1);
        assertTrue(true, "Reward claim should succeed after reveal");
    }

    function test_slashCommitmentPaysFinder() public {
        uint256 penalty = 1 ether;
        uint16 finderRewardBps = 1_000; // 10%

        vm.prank(aggregator);
        swarm.configureTrainerPrivacy(penalty, finderRewardBps);
        vm.prank(aggregator);
        swarm.setTrainerPrivacyMode(true);

        vm.deal(aggregator, 3 ether);
        vm.prank(aggregator);
        swarm.depositAggregatorBond{value: 3 ether}();

        vm.prank(aggregator);
        swarm.startTrainingRound();

        bytes32 nonce = keccak256("nonce");
        bytes32 commitment = keccak256(abi.encodePacked(trainer1, nonce));
        bytes32 modelHash = keccak256("model1");
        uint64 deadline = uint64(block.timestamp + 1 hours);

        vm.prank(aggregator);
        swarm.registerRoundContributionPrivacy(
            1,
            commitment,
            modelHash,
            deadline
        );

        vm.warp(deadline + 1);

        address payable finder = payable(makeAddr("finder"));
        uint256 finderBalanceBefore = finder.balance;

        vm.prank(finder);
        (uint256 slashedPenalty, uint256 finderReward) = swarm
            .slashAggregatorBond(1, commitment);
        assertEq(slashedPenalty, penalty, "Penalty should match configuration");
        assertEq(
            finderReward,
            (penalty * finderRewardBps) / 10_000,
            "Finder reward should match configuration"
        );
        assertEq(
            finder.balance,
            finderBalanceBefore + finderReward,
            "Finder should receive reward"
        );

        (uint256 bondBalance, uint256 bondReserved) = swarm
            .getAggregatorBondState();
        assertEq(bondReserved, 0, "Reserved bond should clear after slash");
        assertEq(
            bondBalance,
            3 ether - penalty,
            "Bond balance should decrease by penalty"
        );

        vm.expectRevert(
            abi.encodeWithSelector(
                RoundTrainerRegistryV2.CommitmentAlreadySlashed.selector,
                1,
                commitment
            )
        );
        vm.prank(finder);
        swarm.slashAggregatorBond(1, commitment);

        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfigAfter = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfigAfter.ttl);
        swarm.updatePhase();

        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationConfigAfter = swarm
                .getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfigAfter.registrationTtl);
        swarm.updatePhase();

        vm.prank(aggregator);
        swarm.revealTrainerCommitment(1, trainer1, abi.encodePacked(nonce));
        assertEq(
            swarm.getTrainerId(1, trainer1),
            1,
            "Trainer should reveal after slash"
        );
    }

    // ============================================================================
    // PHASE TRANSITION TESTS
    // ============================================================================

    function test_fullTrainingCycle() public {
        // Test complete training cycle
        assertTrue(swarm.isIdle(), "Should start in idle");

        // Start training
        vm.prank(aggregator);
        swarm.startTrainingRound();
        assertTrue(swarm.isTraining(), "Should be in training");

        // Register trainers
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));

        // Fast forward to evaluator registration using actual config
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();
        assertTrue(
            swarm.isEvaluation(),
            "Should be in evaluation (registration phase)"
        );

        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);

        // Fast forward to evaluation phase using actual config
        BaseTrainingPhases.EvaluationPhaseConfiguration
            memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();
        assertTrue(
            swarm.isEvaluation(),
            "Should still be in evaluation (evaluation phase)"
        );

        // Register evaluations with proper flow
        vm.startPrank(evaluator1);
        uint256 evalId1 = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId1 = swarm.nthTaskOfNode(1, evalId1 - 1, 0);
        ContributionCalculator contributionCalculator = ContributionCalculator(
            swarm.getContributionCalculator()
        );
        uint256 mask1 = contributionCalculator.getMask(1, taskId1, 2);
        swarm.registerEvaluation(1, taskId1, mask1, keccak256("model1"), 100);
        vm.stopPrank();

        vm.startPrank(evaluator2);
        uint256 evalId2 = swarm.getEvaluatorId(1, evaluator2);
        uint256 taskId2 = swarm.nthTaskOfNode(1, evalId2 - 1, 0);
        uint256 mask2 = contributionCalculator.getMask(1, taskId2, 2);
        swarm.registerEvaluation(1, taskId2, mask2, keccak256("model2"), 200);
        vm.stopPrank();

        // Fast forward to idle using actual config
        vm.warp(block.timestamp + evaluationConfig.ttl);
        swarm.updatePhase();
        assertTrue(swarm.isIdle(), "Should be back in idle");
    }

    // ============================================================================
    // SELECTOR UPDATE TESTS
    // ============================================================================

    function test_updateTrainerSelector() public {
        // Deploy new selector
        AlwaysSampled newSelector = new AlwaysSampled();

        // Update selector
        vm.prank(aggregator);
        swarm.updateTrainerSelector(address(newSelector));

        // Should not revert
        assertTrue(true, "Trainer selector should be updated");
    }

    function test_updateEvaluatorSelector() public {
        // Deploy new selector
        AlwaysSampled newSelector = new AlwaysSampled();

        // Update selector
        vm.prank(aggregator);
        swarm.updateEvaluatorSelector(address(newSelector));

        // Should not revert
        assertTrue(true, "Evaluator selector should be updated");
    }

    // ============================================================================
    // DISTRIBUTION TESTS
    // ============================================================================

    function test_distribute() public {
        address[] memory trainers = new address[](2);
        trainers[0] = trainer1;
        trainers[1] = trainer2;

        uint64[] memory contributions = new uint64[](2);
        contributions[0] = 100;
        contributions[1] = 200;

        vm.prank(aggregator);
        swarm.distribute(1, trainers, contributions);

        // Should not revert
        assertTrue(true, "Distribution should succeed");
    }

    // ============================================================================
    // CERTIFICATE TESTS
    // ============================================================================

    function test_setCertificate() public {
        bytes32 id = keccak256("test-certificate");
        bytes memory value = "test-certificate-data";

        vm.prank(aggregator);
        swarm.setCertificate(id, value);

        // Should not revert
        assertTrue(true, "Certificate should be set");
    }

    // ============================================================================
    // CAN TRAIN TESTS
    // ============================================================================

    function test_canTrain() public view {
        // Test canTrain function
        assertTrue(
            swarm.canTrain(trainer1, 1),
            "Trainer1 should be able to train"
        );
        assertTrue(
            swarm.canTrain(trainer2, 1),
            "Trainer2 should be able to train"
        );
        assertTrue(
            swarm.canTrain(trainer3, 1),
            "Trainer3 should be able to train"
        );
        assertFalse(
            swarm.canTrain(evaluator1, 1),
            "Evaluator1 should not be able to train"
        );
    }

    // ============================================================================
    // EDGE CASES AND ERROR TESTS
    // ============================================================================

    function test_registerRoundContribution_duplicate() public {
        // Start training round
        vm.prank(aggregator);
        swarm.startTrainingRound();

        // Register contribution twice
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));

        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model2"));

        // Should not revert (duplicate registration should be handled)
        assertTrue(true, "Duplicate registration should be handled");
    }

    function test_registerForRoundEvaluation_duplicate() public {
        // Complete setup to evaluator registration phase
        vm.prank(aggregator);
        swarm.startTrainingRound();

        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        vm.prank(trainer2);
        swarm.registerRoundContribution(1, keccak256("model2"));

        // Fast forward to evaluator registration phase using actual config
        BaseTrainingPhases.TrainingPhaseConfiguration
            memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();

        // Register evaluator twice
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);

        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);

        // Should not revert (duplicate registration should be handled)
        assertTrue(true, "Duplicate evaluator registration should be handled");
    }

    // ============================================================================
    // INTERFACE SUPPORT TESTS
    // ============================================================================

    function test_supportsInterface() public view {
        // Test interface support
        assertTrue(
            swarm.supportsInterface(type(IERC165).interfaceId),
            "Should support ERC165"
        );
        assertTrue(
            swarm.supportsInterface(swarm.canTrain.selector),
            "Should support canTrain"
        );
        assertTrue(
            swarm.supportsInterface(swarm.distribute.selector),
            "Should support distribute"
        );
    }

    // ============================================================================
    // GAS OPTIMIZATION TESTS
    // ============================================================================

    function test_startTrainingRound_gasUsage() public {
        uint256 gasStart = gasleft();
        vm.prank(aggregator);
        swarm.startTrainingRound();
        uint256 gasUsed = gasStart - gasleft();

        assertLt(gasUsed, 200000, "Gas usage should be reasonable");
    }

    function test_registerRoundContribution_gasUsage() public {
        vm.prank(aggregator);
        swarm.startTrainingRound();

        uint256 gasStart = gasleft();
        vm.prank(trainer1);
        swarm.registerRoundContribution(1, keccak256("model1"));
        uint256 gasUsed = gasStart - gasleft();

        assertLt(gasUsed, 150000, "Gas usage should be reasonable");
    }
}
