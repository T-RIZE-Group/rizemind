// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {IAccessControl} from "@openzeppelin-contracts-5.2.0/access/IAccessControl.sol";

contract ContributionCalculatorTest is Test {
    ContributionCalculator public implementation;
    ContributionCalculator public calculator;
    address public admin;
    address public user;

    function setUp() public {
        admin = makeAddr("admin");
        user = makeAddr("user");
        
        // Deploy implementation
        implementation = new ContributionCalculator();
        
        // Deploy proxy
        bytes memory initData = abi.encodeWithSelector(
            ContributionCalculator.initialize.selector,
            admin
        );
        
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(implementation),
            initData
        );
        
        calculator = ContributionCalculator(address(proxy));
    }

    function test_initialize() public {
        assertTrue(calculator.hasRole(calculator.DEFAULT_ADMIN_ROLE(), admin));
    }

    function test_registerResult_adminOnly() public {
        vm.startPrank(admin);
        
        uint256 roundId = 1;
        uint256 sampleId = 2;
        uint8 numberOfPlayers = 2;
        uint256 setId = calculator.getMask(roundId, sampleId, numberOfPlayers);
        bytes32 modelHash = keccak256("test_model");
        int256 result = 100;

        calculator.registerResult(roundId, setId, setId, modelHash, result, numberOfPlayers);
        
        // Verify result was stored
        int256 retrievedResult = calculator.getResult(roundId, setId);
        assertEq(retrievedResult, result);
        
        vm.stopPrank();
    }

    function test_registerResult_unauthorized() public {
        vm.startPrank(user);
        
        uint256 roundId = 1;
        uint256 setId = 123;
        bytes32 modelHash = keccak256("test_model");
        int256 result = 100;

        vm.expectRevert(
            abi.encodeWithSelector(
                IAccessControl.AccessControlUnauthorizedAccount.selector,
                user,
                calculator.DEFAULT_ADMIN_ROLE()
            )
        );
        calculator.registerResult(roundId, setId, setId, modelHash, result, 0);
        
        vm.stopPrank();
    }

    function test_calculateShapleyValue() public {
        // First register some results
        vm.startPrank(admin);
        
        uint256 roundId = 1;
        bytes32 modelHash = keccak256("test_model");
        
        // Register result for empty coalition
        calculator.registerResult(roundId, 0, 0, modelHash, 0, 2);
        
        // Register result for single trainer
        calculator.registerResult(roundId, 1, 1, modelHash, 100, 2);
        
        // Register result for both trainers
        calculator.registerResult(roundId, 3, 3, modelHash, 200, 2);
        
        vm.stopPrank();
        
        calculator.calculateContribution(roundId, 0, 2);

    }
}
