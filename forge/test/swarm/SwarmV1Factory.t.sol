// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {SwarmV1} from "@rizemind-contracts/swarm/SwarmV1.sol";
import {SwarmV1Factory} from "@rizemind-contracts/swarm/SwarmV1Factory.sol";
import {SelectorFactory} from "@rizemind-contracts/sampling/SelectorFactory.sol";
import {AlwaysSampled} from "@rizemind-contracts/sampling/AlwaysSampled.sol";
import {RandomSampling} from "@rizemind-contracts/sampling/RandomSampling.sol";

contract SwarmV1FactoryTest is Test {
    SwarmV1 public swarmImpl;
    SwarmV1Factory public swarmFactory;
    SelectorFactory public selectorFactory;
    AlwaysSampled public trainerSelectorImpl;
    RandomSampling public evaluatorSelectorImpl;
    
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    // Selector IDs for testing
    bytes32  TRAINER_SELECTOR_ID;
    bytes32 EVALUATOR_SELECTOR_ID;

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
        
        // Deploy SelectorFactory first
        selectorFactory = new SelectorFactory(aggregator);
        
        // Now we can get the IDs after SelectorFactory is deployed
        (,, string memory version,,,,) = trainerSelectorImpl.eip712Domain();
        TRAINER_SELECTOR_ID = selectorFactory.getID(version);
        (,, string memory version2,,,,) = evaluatorSelectorImpl.eip712Domain();
        EVALUATOR_SELECTOR_ID = selectorFactory.getID(version2);
        
        selectorFactory.registerSelectorImplementation(address(trainerSelectorImpl));
        selectorFactory.registerSelectorImplementation(address(evaluatorSelectorImpl));
        swarmFactory = new SwarmV1Factory(address(swarmImpl), address(selectorFactory));
        vm.stopPrank();
    }

    function testFactoryDeploy() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("hello", "world");
        
        vm.expectEmit(false, false, true, false);
        emit SwarmV1Factory.ContractCreated(address(0), address(0), "hello");
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(params);
        
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
        address proxyAddress = swarmFactory.createSwarm(params);
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

    function testCreateSwarmWithDifferentNames() public {
        // Test creating multiple swarms with different names
        SwarmV1Factory.SwarmParams memory params1 = _createSwarmParams("swarm1", "SWM1");
        SwarmV1Factory.SwarmParams memory params2 = _createSwarmParams("swarm2", "SWM2");
        
        vm.prank(aggregator);
        address proxy1 = swarmFactory.createSwarm(params1);
        vm.prank(aggregator);
        address proxy2 = swarmFactory.createSwarm(params2);
        
        assertTrue(proxy1 != address(0), "First proxy should be created");
        assertTrue(proxy2 != address(0), "Second proxy should be created");
        assertTrue(proxy1 != proxy2, "Proxies should have different addresses");
    }

    function testCreateSwarmWithEmptyInitData() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("empty", "EMP");
        // Set empty init data for selectors
        params.trainerSelector.initData = "";
        params.evaluatorSelector.initData = "";
        
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(params);
        assertTrue(proxyAddress != address(0), "Proxy should be created even with empty init data");
    }

    function testCreateSwarmWithCustomInitData() public {
        SwarmV1Factory.SwarmParams memory params = _createSwarmParams("custom", "CST");
        uint256 targetRatio = 0.5 ether;    
        params.evaluatorSelector.initData = abi.encodeWithSelector(RandomSampling.initialize.selector, targetRatio);
        
        vm.prank(aggregator);
        address proxyAddress = swarmFactory.createSwarm(params);
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
        swarmFactory.createSwarm(params);
        
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

    // Helper function to create SwarmParams for testing
    function _createSwarmParams(
        string memory name,
        string memory symbol
    ) internal view returns (SwarmV1Factory.SwarmParams memory) {
        SwarmV1Factory.SwarmV1Params memory swarmParams = SwarmV1Factory.SwarmV1Params({
            name: name,
            symbol: symbol,
            aggregator: aggregator,
            initialTrainers: trainers,
            initialTrainerSelector: address(0) // This will be set by the factory
        });
        
        SwarmV1Factory.SelectorParams memory trainerSelectorParams = SwarmV1Factory.SelectorParams({
            id: TRAINER_SELECTOR_ID,
            initData: ""
        });
        
        SwarmV1Factory.SelectorParams memory evaluatorSelectorParams = SwarmV1Factory.SelectorParams({
            id: EVALUATOR_SELECTOR_ID,
            initData: ""
        });
        
        return SwarmV1Factory.SwarmParams({
            swarm: swarmParams,
            trainerSelector: trainerSelectorParams,
            evaluatorSelector: evaluatorSelectorParams
        });
    }
}
