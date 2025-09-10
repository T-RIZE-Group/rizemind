// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {AlwaysSampled} from "@rizemind-contracts/sampling/AlwaysSampled.sol";
import {RandomSampling} from "@rizemind-contracts/sampling/RandomSampling.sol";
import {CalculatorFactory} from "@rizemind-contracts/contribution/CalculatorFactory.sol";
import {ContributionCalculator} from "@rizemind-contracts/contribution/ContributionCalculator.sol";

contract SwarmV1FactoryTest is Test {
    SwarmV1 public swarmImpl;
    SwarmV1Factory public swarmFactory;
    SelectorFactory public selectorFactory;
    CalculatorFactory public calculatorFactory;
    ContributionCalculator public calculatorImpl;
    AlwaysSampled public trainerSelectorImpl;
    RandomSampling public evaluatorSelectorImpl;
    
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    // Selector IDs for testing
    bytes32 TRAINER_SELECTOR_ID;
    bytes32 EVALUATOR_SELECTOR_ID;
    bytes32 CALCULATOR_ID;

    bytes32 salt = keccak256(abi.encodePacked("test"));

    function setUp() public {
        aggregator = vm.addr(1);
        trainers.push(vm.addr(2));
        trainers.push(vm.addr(3));
        trainers.push(vm.addr(4));
        nonTrainer = vm.addr(5);
        
        // Deploy the implementation contract (SwarmV1)
        vm.startPrank(aggregator);
        swarmImpl = new SwarmV1();
        
        // Deploy selector implementations
        trainerSelectorImpl = new AlwaysSampled();
        evaluatorSelectorImpl = new RandomSampling();
        
        // Deploy calculator implementation
        calculatorImpl = new ContributionCalculator();
        
        // Deploy SelectorFactory first
        selectorFactory = new SelectorFactory(aggregator);
        
        // Deploy CalculatorFactory
        calculatorFactory = new CalculatorFactory(aggregator);
        
        // Now we can get the IDs after factories are deployed
        (,, string memory version,,,,) = trainerSelectorImpl.eip712Domain();
        TRAINER_SELECTOR_ID = selectorFactory.getID(version);
        (,, string memory version2,,,,) = evaluatorSelectorImpl.eip712Domain();
        EVALUATOR_SELECTOR_ID = selectorFactory.getID(version2);
        (,, string memory version3,,,,) = calculatorImpl.eip712Domain();
        CALCULATOR_ID = calculatorFactory.getID(version3);
        
        selectorFactory.registerSelectorImplementation(address(trainerSelectorImpl));
        selectorFactory.registerSelectorImplementation(address(evaluatorSelectorImpl));
        calculatorFactory.registerCalculatorImplementation(address(calculatorImpl));
        swarmFactory = new SwarmV1Factory(address(swarmImpl), address(selectorFactory));
        swarmFactory.setCalculatorFactory(address(calculatorFactory));
        vm.stopPrank();
    }

    function testFactoryDeploy() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("hello", "world");
        
        vm.expectEmit(false, false, true, false);
        emit SwarmV1Factory.ContractCreated(address(0), address(0), "hello");
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(salt,params);
        
        assertTrue(proxyAddress != address(0), "Proxy should be created");
        
        // Verify the proxy is properly initialized
        // Note: We can't directly call initialize again due to initializer modifier
        // But we can verify the proxy was created successfully
    }

    function testUpdateImplementation() public {
        vm.prank(aggregator);
        SwarmV1 newImpl = new SwarmV1();
        vm.prank(aggregator);
        swarmFactory.updateImplementation(address(newImpl));
        
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("hello", "world");
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(salt, params);
        assertTrue(proxyAddress != address(0), "Proxy should be created with new implementation");
    }

    function testUpdateImplementationProtected() public {
        vm.prank(trainers[0]);
        vm.expectRevert();
        swarmFactory.updateImplementation(trainers[0]);
    }

    function testGetImplementation() public view {
        assertEq(
            swarmFactory.getImplementation(),
            address(swarmImpl),
            "getImplementation should return the correct address"
        );
    }

    function testUpdateImplementationZeroAddressReverts() public {
        vm.prank(aggregator);
        vm.expectRevert(bytes("implementation cannot be null"));
        swarmFactory.updateImplementation(address(0));
    }

    function testCreateSwarmWithEmptyInitData() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("empty", "EMP");
        // Set empty init data for selectors
        params.trainerSelector.initData = "";
        params.evaluatorSelector.initData = "";
        
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(salt, params);
        assertTrue(proxyAddress != address(0), "Proxy should be created even with empty init data");
    }

    function testCreateSwarmWithCustomInitData() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("custom", "CST");
        uint256 targetRatio = 0.5 ether;    
        params.evaluatorSelector.initData = abi.encodeWithSelector(RandomSampling.initialize.selector, targetRatio);
        
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(salt, params);
        assertTrue(proxyAddress != address(0), "Proxy should be created with custom init data");
        SwarmV1 swarm = SwarmV1(proxyAddress);
        RandomSampling evaluatorSelector = RandomSampling(swarm.getEvaluatorSelector());
        assertEq(evaluatorSelector.getTargetRatio(), targetRatio);
    }

    function testCreateSwarmEmitsCorrectEvents() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("test", "TST");
        
        // Capture the event and verify the parameters we care about
        vm.recordLogs();
        vm.prank(aggregator);
        swarmFactory.createSwarm(salt, params);
        
        Vm.Log[] memory logs = vm.getRecordedLogs();
        assertTrue(logs.length > 0, "Should emit at least one event");
        
        // Verify the event was emitted with correct parameters
        bytes32 eventSignature = keccak256("ContractCreated(address,address,string)");
        bool eventFound = false;
        
        for (uint i = 0; i < logs.length; i++) {
            if (logs[i].topics[0] == eventSignature) {
                // Check that the proxy address is correct (first indexed parameter)
                address proxyAddress = address(uint160(uint256(logs[i].topics[1])));
                assertTrue(proxyAddress != address(0), "Proxy address should not be zero");
                
                // Check that the owner is correct (second indexed parameter)
                address owner = address(uint160(uint256(logs[i].topics[2])));
                assertEq(owner, aggregator, "Event owner should match aggregator");
                
                // Check that the name is correct (third parameter in data)
                string memory name = abi.decode(logs[i].data, (string));
                assertEq(name, "test", "Event name should match");
                
                eventFound = true;
                break;
            }
        }
        
        assertTrue(eventFound, "ContractCreated event should be emitted");
    }

    function testTwoSelectorsDontCollideDueToSalt() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("hello", "world");
        params.trainerSelector.id = TRAINER_SELECTOR_ID;
        params.evaluatorSelector.id = TRAINER_SELECTOR_ID;
        
        vm.expectEmit(false, false, true, false);
        emit SwarmV1Factory.ContractCreated(address(0), address(0), "hello");
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(salt, params);
        
        assertTrue(proxyAddress != address(0), "Proxy should be created");
        
    }

    // Helper function to create SwarmParams for testing
    function _createSwarmParams(
        string memory name,
        string memory symbol
    ) internal view returns (SwarmV1Factory.SwarmParams memory) {
        SwarmV1Factory.SwarmV1Params memory swarmParams = SwarmV1Factory.SwarmV1Params({
            name: name,
            symbol: symbol,
            aggregator: aggregator,
            trainers: trainers
        });
        
        SwarmV1Factory.SelectorParams memory trainerSelectorParams = SwarmV1Factory.SelectorParams({
            id: TRAINER_SELECTOR_ID,
            initData: ""
        });
        
        SwarmV1Factory.SelectorParams memory evaluatorSelectorParams = SwarmV1Factory.SelectorParams({
            id: EVALUATOR_SELECTOR_ID,
            initData: ""
        });
        
        SwarmV1Factory.CalculatorParams memory calculatorParams = SwarmV1Factory.CalculatorParams({
            id: CALCULATOR_ID,
            initData: abi.encodeWithSelector(ContributionCalculator.initialize.selector, aggregator, 2)
        });
        
        return SwarmV1Factory.SwarmParams({
            swarm: swarmParams,
            trainerSelector: trainerSelectorParams,
            evaluatorSelector: evaluatorSelectorParams,
            calculatorFactory: calculatorParams
        });
    }

    // ============================================================================
    // CALCULATOR FACTORY TESTS
    // ============================================================================

    function testGetCalculatorFactory() public view {
        // Should be set to calculatorFactory address from setUp
        assertEq(swarmFactory.getCalculatorFactory(), address(calculatorFactory), "Calculator factory should be set correctly");
    }

    function testSetCalculatorFactory() public {
        // Deploy a new calculator factory for this test
        vm.prank(aggregator);
        CalculatorFactory newCalculatorFactory = new CalculatorFactory(aggregator);
        
        vm.prank(aggregator);
        swarmFactory.setCalculatorFactory(address(newCalculatorFactory));
        
        assertEq(swarmFactory.getCalculatorFactory(), address(newCalculatorFactory), "Calculator factory should be set correctly");
    }

    function testSetCalculatorFactoryEmitsEvent() public {
        // Deploy a new calculator factory for this test
        vm.prank(aggregator);
        CalculatorFactory newCalculatorFactory = new CalculatorFactory(aggregator);
        
        vm.expectEmit(true, true, false, false);
        emit SwarmV1Factory.CalculatorFactoryUpdated(address(calculatorFactory), address(newCalculatorFactory));
        
        vm.prank(aggregator);
        swarmFactory.setCalculatorFactory(address(newCalculatorFactory));
    }

    function testSetCalculatorFactoryProtected() public {
        vm.prank(trainers[0]);
        vm.expectRevert();
        swarmFactory.setCalculatorFactory(address(calculatorFactory));
    }

    function testSetCalculatorFactoryZeroAddressReverts() public {
        vm.prank(aggregator);
        vm.expectRevert(bytes("calculator factory cannot be null"));
        swarmFactory.setCalculatorFactory(address(0));
    }

    function testUpdateCalculatorFactory() public {
        // First set a calculator factory
        vm.prank(aggregator);
        swarmFactory.setCalculatorFactory(address(calculatorFactory));
        
        // Deploy a new calculator factory
        vm.prank(aggregator);
        CalculatorFactory newCalculatorFactory = new CalculatorFactory(aggregator);
        
        // Update to new calculator factory
        vm.expectEmit(true, true, false, false);
        emit SwarmV1Factory.CalculatorFactoryUpdated(address(calculatorFactory), address(newCalculatorFactory));
        
        vm.prank(aggregator);
        swarmFactory.setCalculatorFactory(address(newCalculatorFactory));
        
        assertEq(swarmFactory.getCalculatorFactory(), address(newCalculatorFactory), "Calculator factory should be updated correctly");
    }

    function testGetSwarmAddressMatchesCreateSwarm() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("test", "TST");
        
        // Get the predicted address before creating
        address predictedAddress = swarmFactory.getSwarmAddress(salt);
        
        // Create the swarm
        vm.prank(aggregator);
        address createdAddress = swarmFactory.createSwarm(salt, params);
        
        // Verify both addresses match
        assertEq(predictedAddress, createdAddress, "getSwarmAddress should return the same address as createSwarm");
        assertTrue(createdAddress != address(0), "Created address should not be zero");
    }
}
