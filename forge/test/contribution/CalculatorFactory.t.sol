// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {CalculatorFactory} from "../../src/contribution/CalculatorFactory.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";

contract CalculatorFactoryTest is Test {
    CalculatorFactory public factory;
    ContributionCalculator public implementation;
    address public owner;
    address public user;

    function setUp() public {
        owner = makeAddr("owner");
        user = makeAddr("user");
        
        // Deploy factory
        vm.prank(owner);
        factory = new CalculatorFactory(owner);
        
        // Deploy implementation
        implementation = new ContributionCalculator();
    }

    function test_initialize() public {
        assertEq(factory.owner(), owner);
    }

    function test_registerCalculatorImplementation() public {
        vm.startPrank(owner);
        
        factory.registerCalculatorImplementation(address(implementation));
        
        bytes32 id = factory.getID("contribution-calculator-v1.0.0");
        assertTrue(factory.isCalculatorRegistered(id));
        assertEq(factory.getCalculatorImplementation(id), address(implementation));
        
        vm.stopPrank();
    }

    function test_registerCalculatorImplementation_unauthorized() public {
        vm.startPrank(user);
        
        vm.expectRevert();
        factory.registerCalculatorImplementation(address(implementation));
        
        vm.stopPrank();
    }

    function test_createCalculator() public {
        vm.startPrank(owner);
        
        // First register the implementation
        factory.registerCalculatorImplementation(address(implementation));
        
        bytes32 id = factory.getID("contribution-calculator-v1.0.0");
        bytes32 salt = keccak256("test-salt");
        address initialAdmin = makeAddr("initialAdmin");
        
        // Create calculator instance
        address instance = factory.createCalculator(id, salt, initialAdmin);
        
        assertTrue(instance != address(0));
        
        // Verify the instance is a proxy pointing to the implementation
        ContributionCalculator calculator = ContributionCalculator(instance);
        assertTrue(calculator.hasRole(calculator.DEFAULT_ADMIN_ROLE(), initialAdmin));
        
        vm.stopPrank();
    }

    function test_createCalculator_notRegistered() public {
        bytes32 id = factory.getID("non-existent-version");
        bytes32 salt = keccak256("test-salt");
        address initialAdmin = makeAddr("initialAdmin");
        
        vm.expectRevert(CalculatorFactory.CalculatorImplementationNotFound.selector);
        factory.createCalculator(id, salt, initialAdmin);
    }

    function test_removeCalculatorImplementation() public {
        vm.startPrank(owner);
        
        // First register the implementation
        factory.registerCalculatorImplementation(address(implementation));
        
        bytes32 id = factory.getID("contribution-calculator-v1.0.0");
        assertTrue(factory.isCalculatorRegistered(id));
        
        // Remove the implementation
        factory.removeCalculatorImplementation(id);
        
        assertFalse(factory.isCalculatorRegistered(id));
        assertEq(factory.getCalculatorImplementation(id), address(0));
        
        vm.stopPrank();
    }

    function test_supportsInterface() public {
        assertTrue(factory.supportsInterface(0x01ffc9a7)); // IERC165
        assertFalse(factory.supportsInterface(0xffffffff)); // Random interface
    }
}
