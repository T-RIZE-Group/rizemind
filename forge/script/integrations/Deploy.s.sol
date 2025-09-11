// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Script.sol";
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

contract DeployAll is Script {
    // Factory contracts
    address public selectorFactory;
    address public calculatorFactory;
    address public accessControlFactory;
    address public compensationFactory;
    address public swarmV1Factory;
    
    // Implementation contracts
    address public alwaysSampledImpl;
    address public randomSamplingImpl;
    address public contributionCalculatorImpl;
    address public baseAccessControlImpl;
    address public simpleMintCompensationImpl;
    address public swarmV1Impl;
    
    // Factory IDs
    bytes32 public trainerSelectorId;
    bytes32 public evaluatorSelectorId;
    bytes32 public calculatorId;
    bytes32 public accessControlId;
    bytes32 public compensationId;

    function run() external {
        address owner = vm.envAddress("OWNER");
        require(owner != address(0), "OWNER environment variable not set");
        
        console.log("=== Starting Complete Federated Learning Deployment ===");
        console.log("Owner:", owner);
        console.log("Chain ID:", block.chainid);
        console.log("");

        vm.startBroadcast();

        // Step 1: Deploy all factory contracts
        console.log("Step 1: Deploying Factory Contracts...");
        _deployFactories(owner);

        vm.stopBroadcast();
        vm.startBroadcast();
        
        // Step 2: Deploy all implementation contracts
        console.log("Step 2: Deploying Implementation Contracts...");
        _deployImplementations();
        vm.stopBroadcast();
        vm.startBroadcast();
        
        // Step 3: Register implementations with factories
        console.log("Step 3: Registering Implementations with Factories...");
        _registerImplementations();
        vm.stopBroadcast();
        vm.startBroadcast();        
        // Step 4: Deploy main SwarmV1Factory
        console.log("Step 4: Deploying Main SwarmV1Factory...");
        _deploySwarmV1Factory();
        vm.stopBroadcast();
        vm.startBroadcast();
        // Step 5: Deploy a test swarm instance
        console.log("Step 5: Deploying Test Swarm Instance...");
        _deployTestSwarm();

        vm.stopBroadcast();

        // Step 6: Print deployment summary
        _printDeploymentSummary();
    }

    function _deployFactories(address owner) internal {
        // Deploy SelectorFactory
        SelectorFactory selectorFactoryContract = new SelectorFactory(owner);
        selectorFactory = address(selectorFactoryContract);
        console.log("  [OK] SelectorFactory deployed at:", selectorFactory);

        // Deploy CalculatorFactory
        CalculatorFactory calculatorFactoryContract = new CalculatorFactory(owner);
        calculatorFactory = address(calculatorFactoryContract);
        console.log("  [OK] CalculatorFactory deployed at:", calculatorFactory);

        // Deploy AccessControlFactory
        AccessControlFactory accessControlFactoryContract = new AccessControlFactory(owner);
        accessControlFactory = address(accessControlFactoryContract);
        console.log("  [OK] AccessControlFactory deployed at:", accessControlFactory);

        // Deploy CompensationFactory
        CompensationFactory compensationFactoryContract = new CompensationFactory(owner);
        compensationFactory = address(compensationFactoryContract);
        console.log("  [OK] CompensationFactory deployed at:", compensationFactory);
    }

    function _deployImplementations() internal {
        // Deploy SwarmV1 implementation
        SwarmV1 swarmV1ImplContract = new SwarmV1();
        swarmV1Impl = address(swarmV1ImplContract);
        console.log("  [OK] SwarmV1 implementation deployed at:", swarmV1Impl);

        // Deploy selector implementations
        AlwaysSampled alwaysSampledImplContract = new AlwaysSampled();
        alwaysSampledImpl = address(alwaysSampledImplContract);
        console.log("  [OK] AlwaysSampled implementation deployed at:", alwaysSampledImpl);

        RandomSampling randomSamplingImplContract = new RandomSampling();
        randomSamplingImpl = address(randomSamplingImplContract);
        console.log("  [OK] RandomSampling implementation deployed at:", randomSamplingImpl);

        // Deploy calculator implementation
        ContributionCalculator contributionCalculatorImplContract = new ContributionCalculator();
        contributionCalculatorImpl = address(contributionCalculatorImplContract);
        console.log("  [OK] ContributionCalculator implementation deployed at:", contributionCalculatorImpl);

        // Deploy access control implementation
        BaseAccessControl baseAccessControlImplContract = new BaseAccessControl();
        baseAccessControlImpl = address(baseAccessControlImplContract);
        console.log("  [OK] BaseAccessControl implementation deployed at:", baseAccessControlImpl);

        // Deploy compensation implementation
        SimpleMintCompensation simpleMintCompensationImplContract = new SimpleMintCompensation();
        simpleMintCompensationImpl = address(simpleMintCompensationImplContract);
        console.log("  [OK] SimpleMintCompensation implementation deployed at:", simpleMintCompensationImpl);
    }

    function _registerImplementations() internal {
        // Get EIP712 domain versions and create IDs
        (,, string memory version1,,,,) = AlwaysSampled(alwaysSampledImpl).eip712Domain();
        trainerSelectorId = SelectorFactory(selectorFactory).getID(version1);
        
        (,, string memory version2,,,,) = RandomSampling(randomSamplingImpl).eip712Domain();
        evaluatorSelectorId = SelectorFactory(selectorFactory).getID(version2);
        
        (,, string memory version3,,,,) = ContributionCalculator(contributionCalculatorImpl).eip712Domain();
        calculatorId = CalculatorFactory(calculatorFactory).getID(version3);
        
        (,, string memory version4,,,,) = BaseAccessControl(baseAccessControlImpl).eip712Domain();
        accessControlId = AccessControlFactory(accessControlFactory).getID(version4);
        
        (,, string memory version5,,,,) = SimpleMintCompensation(simpleMintCompensationImpl).eip712Domain();
        compensationId = CompensationFactory(compensationFactory).getID(version5);

        // Register selector implementations
        SelectorFactory(selectorFactory).registerSelectorImplementation(alwaysSampledImpl);
        SelectorFactory(selectorFactory).registerSelectorImplementation(randomSamplingImpl);
        console.log("  [OK] Selector implementations registered");

        // Register calculator implementation
        CalculatorFactory(calculatorFactory).registerCalculatorImplementation(contributionCalculatorImpl);
        console.log("  [OK] Calculator implementation registered");

        // Register access control implementation
        AccessControlFactory(accessControlFactory).registerAccessControlImplementation(baseAccessControlImpl);
        console.log("  [OK] AccessControl implementation registered");

        // Register compensation implementation
        CompensationFactory(compensationFactory).registerCompensationImplementation(simpleMintCompensationImpl);
        console.log("  [OK] Compensation implementation registered");
    }

    function _deploySwarmV1Factory() internal {
        SwarmV1Factory swarmV1FactoryContract = new SwarmV1Factory(
            swarmV1Impl,
            selectorFactory,
            calculatorFactory,
            accessControlFactory,
            compensationFactory
        );
        swarmV1Factory = address(swarmV1FactoryContract);
        console.log("  [OK] SwarmV1Factory deployed at:", swarmV1Factory);
    }

    function _deployTestSwarm() internal {
        // Create test parameters
        address[] memory testTrainers = new address[](3);
        testTrainers[0] = vm.addr(1);
        testTrainers[1] = vm.addr(2);
        testTrainers[2] = vm.addr(3);

        address[] memory testEvaluators = new address[](2);
        testEvaluators[0] = vm.addr(4);
        testEvaluators[1] = vm.addr(5);

        address testAggregator = vm.addr(6);
        address testSwarmAddress = vm.addr(7);

        // Create SwarmParams
        SwarmV1Factory.SwarmParams memory params = SwarmV1Factory.SwarmParams({
            swarm: SwarmV1Factory.SwarmV1Params({
                name: "TestFederatedLearningSwarm"
            }),
            trainerSelector: SwarmV1Factory.SelectorParams({
                id: trainerSelectorId,
                initData: abi.encodeWithSelector(AlwaysSampled.initialize.selector)
            }),
            evaluatorSelector: SwarmV1Factory.SelectorParams({
                id: evaluatorSelectorId,
                initData: abi.encodeWithSelector(RandomSampling.initialize.selector, 1 ether) // 100% selection rate
            }),
            calculatorFactory: SwarmV1Factory.CalculatorParams({
                id: calculatorId,
                initData: abi.encodeWithSelector(ContributionCalculator.initialize.selector, testSwarmAddress, 2)
            }),
            accessControl: SwarmV1Factory.AccessControlParams({
                id: accessControlId,
                initData: abi.encodeWithSelector(BaseAccessControl.initialize.selector, testAggregator, testTrainers, testEvaluators)
            }),
            compensation: SwarmV1Factory.CompensationParams({
                id: compensationId,
                initData: abi.encodeWithSelector(SimpleMintCompensation.initialize.selector, "TestToken", "TST", 1000 ether, testAggregator, testSwarmAddress)
            }),
            trainingPhaseConfiguration: BaseTrainingPhases.TrainingPhaseConfiguration({
                ttl: 1000
            }),
            evaluationPhaseConfiguration: BaseTrainingPhases.EvaluationPhaseConfiguration({
                ttl: 1000,
                registrationTtl: 1000
            })
        });

        // Deploy test swarm
        address testSwarm = SwarmV1Factory(swarmV1Factory).createSwarm(keccak256("test-integration"), params);
        console.log("  [OK] Test Swarm deployed at:", testSwarm);
    }

    function _printDeploymentSummary() internal view {
        console.log("");
        console.log("=== DEPLOYMENT SUMMARY ===");
        console.log("");
        console.log("[FACTORY] FACTORY CONTRACTS:");
        console.log("  SelectorFactory:        ", selectorFactory);
        console.log("  CalculatorFactory:      ", calculatorFactory);
        console.log("  AccessControlFactory:   ", accessControlFactory);
        console.log("  CompensationFactory:    ", compensationFactory);
        console.log("  SwarmV1Factory:         ", swarmV1Factory);
        console.log("");
        console.log("[IMPL] IMPLEMENTATION CONTRACTS:");
        console.log("  SwarmV1:                ", swarmV1Impl);
        console.log("  AlwaysSampled:          ", alwaysSampledImpl);
        console.log("  RandomSampling:         ", randomSamplingImpl);
        console.log("  ContributionCalculator: ", contributionCalculatorImpl);
        console.log("  BaseAccessControl:      ", baseAccessControlImpl);
        console.log("  SimpleMintCompensation: ", simpleMintCompensationImpl);
        console.log("");
        console.log("[ID] FACTORY IDs:");
        console.log("  Trainer Selector ID:    ", vm.toString(trainerSelectorId));
        console.log("  Evaluator Selector ID:  ", vm.toString(evaluatorSelectorId));
        console.log("  Calculator ID:          ", vm.toString(calculatorId));
        console.log("  Access Control ID:      ", vm.toString(accessControlId));
        console.log("  Compensation ID:        ", vm.toString(compensationId));
        console.log("");
        console.log("[SUCCESS] All contracts deployed successfully!");
        console.log("");
        console.log("Next steps:");
        console.log("  1. Verify contracts on block explorer");
        console.log("  2. Run tests: forge test");
        console.log("  3. Use SwarmV1Factory to create new swarms");
        console.log("");
    }

    // Helper functions to get deployed addresses
    function getSelectorFactory() external view returns (address) {
        return selectorFactory;
    }

    function getCalculatorFactory() external view returns (address) {
        return calculatorFactory;
    }

    function getAccessControlFactory() external view returns (address) {
        return accessControlFactory;
    }

    function getCompensationFactory() external view returns (address) {
        return compensationFactory;
    }

    function getSwarmV1Factory() external view returns (address) {
        return swarmV1Factory;
    }

    function getAllFactoryAddresses() external view returns (
        address _selectorFactory,
        address _calculatorFactory,
        address _accessControlFactory,
        address _compensationFactory,
        address _swarmV1Factory
    ) {
        return (selectorFactory, calculatorFactory, accessControlFactory, compensationFactory, swarmV1Factory);
    }
}
