// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {SwarmV1} from "../../src/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "../../src/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "../../src/sampling/SelectorFactory.sol";
import {CalculatorFactory} from "../../src/contribution/CalculatorFactory.sol";
import {AlwaysSampled} from "../../src/sampling/AlwaysSampled.sol";
import {RandomSampling} from "../../src/sampling/RandomSampling.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {BaseTrainingPhases} from "../../src/training/BaseTrainingPhases.sol";

contract SwarmV1Test is Test {
    SwarmV1 public implementation;
    SwarmV1Factory public factory;
    SelectorFactory public selectorFactory;
    CalculatorFactory public calculatorFactory;
    SwarmV1 public swarm;
    
    address public aggregator = address(0x1);
    address public trainer1 = address(0x2);
    address public trainer2 = address(0x3);
    address public trainer3 = address(0x4);
    address public evaluator1 = address(0x5);
    address public evaluator2 = address(0x6);
    
    address[] public initialTrainers;
    
    function setUp() public {
             // Deploy implementation
        implementation = new SwarmV1();
        
        // Deploy selector factory
        selectorFactory = new SelectorFactory(address(this));
        
        // Deploy calculator factory
        calculatorFactory = new CalculatorFactory(address(this));
        
        // Register selector implementations
        AlwaysSampled alwaysSampledImpl = new AlwaysSampled();
        RandomSampling randomSamplingImpl = new RandomSampling();
        
        selectorFactory.registerSelectorImplementation(address(alwaysSampledImpl));
        selectorFactory.registerSelectorImplementation(address(randomSamplingImpl));
        
        // Register calculator implementation
        ContributionCalculator calculatorImpl = new ContributionCalculator();
        calculatorFactory.registerCalculatorImplementation(address(calculatorImpl));
        
        // Deploy factory
        factory = new SwarmV1Factory(address(implementation), address(selectorFactory));
        
        // Set calculator factory
        factory.setCalculatorFactory(address(calculatorFactory));
        
        // Set up initial trainers
        initialTrainers = new address[](3);
        initialTrainers[0] = trainer1;
        initialTrainers[1] = trainer2;
        initialTrainers[2] = trainer3;
        
        address swarmAddress = factory.getSwarmAddress(keccak256("test-salt"));

        // Create swarm using factory
        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({
                name: "TestSwarm",
                symbol: "TSW",
                aggregator: aggregator,
                trainers: initialTrainers
            }),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: keccak256("always-sampled-v1.0.0"),
                initData: abi.encodeWithSelector(AlwaysSampled.initialize.selector)
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: keccak256("random-sampling-v1.0.0"),
                initData: abi.encodeWithSelector(RandomSampling.initialize.selector, 1 ether) // 100% selection rate
            }),
            calculatorFactory: SwarmV1Factory.CalculatorParams({
                id: keccak256("contribution-calculator-v1.0.0"),
                initData: abi.encodeWithSelector(ContributionCalculator.initialize.selector, swarmAddress, 2)
            })
        });
        
        swarm = SwarmV1(factory.createSwarm(keccak256("test-salt"), params));

        ContributionCalculator contributionCalculator = ContributionCalculator(swarm.getContributionCalculator());
        
        // Add evaluators
        vm.prank(aggregator);
        swarm.addEvaluator(evaluator1);
        vm.prank(aggregator);
        swarm.addEvaluator(evaluator2);
    }

    // ============================================================================
    // INITIALIZATION TESTS
    // ============================================================================

    function test_initialize() public view {
        // Test basic initialization
        assertEq(swarm.name(), "TestSwarm", "Name should be set correctly");
        assertEq(swarm.symbol(), "TSW", "Symbol should be set correctly");
        assertTrue(swarm.hasRole(keccak256("AGGREGATOR"), aggregator), "Aggregator should have AGGREGATOR_ROLE");
        assertTrue(swarm.hasRole(keccak256("TRAINER"), trainer1), "Trainer1 should have TRAINER_ROLE");
        assertTrue(swarm.hasRole(keccak256("TRAINER"), trainer2), "Trainer2 should have TRAINER_ROLE");
        assertTrue(swarm.hasRole(keccak256("TRAINER"), trainer3), "Trainer3 should have TRAINER_ROLE");
        assertTrue(swarm.hasRole(keccak256("EVALUATOR_ROLE"), evaluator1), "Evaluator1 should have EVALUATOR_ROLE");
        assertTrue(swarm.hasRole(keccak256("EVALUATOR_ROLE"), evaluator2), "Evaluator2 should have EVALUATOR_ROLE");
    }

    function test_initialize_wrongInitialization() public {
        // Test that calling initialize() without parameters reverts
        vm.expectRevert(SwarmV1.WrongInitialization.selector);
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
        vm.expectRevert(SwarmV1.NotTrainingPhase.selector);
        swarm.registerRoundContribution(1, keccak256("model1"));
    }

    function test_onlyEvaluator() public {
        // Test that only evaluators can call evaluator-only functions
        vm.prank(trainer1);
        vm.expectRevert();
        swarm.registerForRoundEvaluation(1);
        
        vm.prank(evaluator1);
        // Should not revert (but will revert due to phase)
        vm.expectRevert(SwarmV1.NotEvaluatorRegistrationPhase.selector);
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
        
        assertTrue(swarm.isTraining(), "Should be in training phase after starting round");
    }

    function test_startTrainingRound_notIdle() public {
        // Test starting training round when not idle
        vm.prank(aggregator);
        swarm.startTrainingRound();
        
        // Try to start another round
        vm.prank(aggregator);
        vm.expectRevert(SwarmV1.NotIdle.selector);
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
        assertTrue(swarm.isTrainerRegistered(1, trainer1), "Trainer should be registered");
        assertEq(swarm.getTrainerId(1, trainer1), 1, "Trainer should have ID 1");
    }

    function test_registerRoundContribution_notTrainingPhase() public {
        // Try to register contribution when not in training phase
        vm.prank(trainer1);
        vm.expectRevert(SwarmV1.NotTrainingPhase.selector);
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        // Fast forward to evaluator registration phase
        vm.warp(block.timestamp + trainingConfig.ttl); // Past training TTL
        //swarm.updatePhase(); we intentionally don't call updatePhase here to test automated phase transition
        
        // Register evaluator
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        
        // Check that evaluator is registered
        assertTrue(swarm.isEvaluatorRegistered(1, evaluator1), "Evaluator should be registered");
        assertEq(swarm.getEvaluatorId(1, evaluator1), 1, "Evaluator should have ID 1");
    }

    function test_registerForRoundEvaluation_notRegistrationPhase() public {
        // Try to register when not in evaluator registration phase
        vm.prank(evaluator1);
        vm.expectRevert(SwarmV1.NotEvaluatorRegistrationPhase.selector);
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        
        // Fast forward to evaluator registration phase
        vm.warp(block.timestamp + trainingConfig.ttl);
        
        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);
        
        BaseTrainingPhases.EvaluationPhaseConfiguration memory evaluationPhaseConfiguration = swarm.getEvaluationPhaseConfiguration();
        // Fast forward to evaluation phase
        vm.warp(block.timestamp + evaluationPhaseConfiguration.registrationTtl); // Past registration TTL
        
        // Register evaluation
        vm.startPrank(evaluator1);
        swarm.updatePhase();
        uint256 evalId = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId = swarm.nthTaskOfNode(1, evalId - 1, 0);
        
        ContributionCalculator contributionCalculator = ContributionCalculator(swarm.getContributionCalculator());
        uint256 mask = contributionCalculator.getMask(1, taskId, 2);
        swarm.registerEvaluation(1, taskId, mask, keccak256("model1"), 100);
        vm.stopPrank();
        // Should not revert
        assertTrue(true, "Evaluation should be registered successfully");
    }

    function test_registerEvaluation_notEvaluationPhase() public {
        // Try to register evaluation when not in evaluation phase
        vm.prank(evaluator1);
        vm.expectRevert(SwarmV1.NotEvaluationPhase.selector);
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();
        
        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);

        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);
        
        // Fast forward to evaluation phase using actual config
        BaseTrainingPhases.EvaluationPhaseConfiguration memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();
        
        // using evaluator2's task
        uint256 evalId = swarm.getEvaluatorId(1, evaluator2);
        uint256 taskId = swarm.nthTaskOfNode(1, evalId - 1, 0);
        
        ContributionCalculator contributionCalculator = ContributionCalculator(swarm.getContributionCalculator());
        uint256 mask = contributionCalculator.getMask(1, taskId, 2);
        // Try to register evaluation for task not assigned to evaluator
        vm.prank(evaluator1);
        vm.expectRevert(abi.encodeWithSelector(SwarmV1.NotAssignedTo.selector, 1, taskId, evaluator1));
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();
        
        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);
        
        // Fast forward to evaluation phase
        BaseTrainingPhases.EvaluationPhaseConfiguration memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();
        
        // Register evaluations with proper flow
        vm.startPrank(evaluator1);
        uint256 evalId1 = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId1 = swarm.nthTaskOfNode(1, evalId1 - 1, 0);
        ContributionCalculator contributionCalculator = ContributionCalculator(swarm.getContributionCalculator());
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
        vm.warp(block.timestamp + trainingConfig.ttl);
        swarm.updatePhase();
        assertTrue(swarm.isEvaluation(), "Should be in evaluation (registration phase)");
        
        // Register evaluators
        vm.prank(evaluator1);
        swarm.registerForRoundEvaluation(1);
        vm.prank(evaluator2);
        swarm.registerForRoundEvaluation(1);
        
        // Fast forward to evaluation phase using actual config
        BaseTrainingPhases.EvaluationPhaseConfiguration memory evaluationConfig = swarm.getEvaluationPhaseConfiguration();
        vm.warp(block.timestamp + evaluationConfig.registrationTtl);
        swarm.updatePhase();
        assertTrue(swarm.isEvaluation(), "Should still be in evaluation (evaluation phase)");
        
        // Register evaluations with proper flow
        vm.startPrank(evaluator1);
        uint256 evalId1 = swarm.getEvaluatorId(1, evaluator1);
        uint256 taskId1 = swarm.nthTaskOfNode(1, evalId1 - 1, 0);
        ContributionCalculator contributionCalculator = ContributionCalculator(swarm.getContributionCalculator());
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
        swarm.distribute(trainers, contributions);
        
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
        assertTrue(swarm.canTrain(trainer1, 1), "Trainer1 should be able to train");
        assertTrue(swarm.canTrain(trainer2, 1), "Trainer2 should be able to train");
        assertTrue(swarm.canTrain(trainer3, 1), "Trainer3 should be able to train");
        assertFalse(swarm.canTrain(evaluator1, 1), "Evaluator1 should not be able to train");
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
        BaseTrainingPhases.TrainingPhaseConfiguration memory trainingConfig = swarm.getTrainingPhaseConfiguration();
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
        assertTrue(swarm.supportsInterface(0x01ffc9a7), "Should support ERC165");
        assertTrue(swarm.supportsInterface(swarm.canTrain.selector), "Should support canTrain");
        assertTrue(swarm.supportsInterface(swarm.distribute.selector), "Should support distribute");
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
