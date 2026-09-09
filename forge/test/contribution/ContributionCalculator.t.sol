// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test, console} from "forge-std/Test.sol";
import {ContributionCalculator} from "../../src/contribution/ContributionCalculator.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {IAccessControl} from "@openzeppelin-contracts-5.2.0/access/IAccessControl.sol";

contract ContributionCalculatorTest is Test {
    ContributionCalculator public implementation;
    ContributionCalculator public calculator;
    address public admin;
    address public user;

    uint8 internal constant DEBUG_NUM_TRAINERS = 13;
    uint256 internal constant DEBUG_ROUND_ID = 999;
    uint256 internal constant DEBUG_GAS_LIMIT = 60_000_000;

    function setUp() public {
        admin = makeAddr("admin");
        user = makeAddr("user");

        implementation = new ContributionCalculator();

        bytes memory initData = abi.encodeWithSelector(
            ContributionCalculator.initialize.selector,
            admin,
            2
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

        calculator.registerResult(
            roundId,
            sampleId,
            setId,
            modelHash,
            result,
            numberOfPlayers
        );

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

        int256 contribution = calculator.calculateContribution(roundId, 0, 2);
        assertEq(contribution, 100);
    }

    function _debugResultForMask(uint256 mask) internal pure returns (int256) {
        return int256(mask * 100);
    }

    function _registerAvailableMasks(
        uint256 roundId,
        uint8 numberOfPlayers,
        uint256 requestedSamples,
        bytes32 modelHash
    ) internal returns (uint256 registeredSamples) {
        calculator.setEvaluationsRequired(roundId, requestedSamples);

        for (uint256 sampleId = 0; sampleId < requestedSamples; ++sampleId) {
            try calculator.getMask(roundId, sampleId, numberOfPlayers) returns (
                uint256 setId
            ) {
                calculator.registerResult(
                    roundId,
                    sampleId,
                    setId,
                    modelHash,
                    _debugResultForMask(setId),
                    numberOfPlayers
                );
                registeredSamples += 1;
            } catch {
                console.log("  samplerStoppedAt", sampleId);
                break;
            }
        }

        calculator.setEvaluationsRequired(roundId, registeredSamples);
    }

    /// @notice Change DEBUG_NUM_TRAINERS and rerun this single test to inspect
    ///         how full 2^N coalition registration behaves during calculateContribution.
    function test_calculateShapleyValue_fullCoalitions_debug() public {
        uint8 numberOfPlayers = DEBUG_NUM_TRAINERS;
        uint256 roundId = DEBUG_ROUND_ID;
        uint256 totalCoalitions = calculator.getTotalEvaluations(
            roundId,
            numberOfPlayers
        );
        bytes32 modelHash = keccak256("debug_shapley_model");

        console.log("debug full coalitions test");
        console.log("  numberOfPlayers", uint256(numberOfPlayers));
        console.log("  requestedSamples", totalCoalitions);

        vm.startPrank(admin);
        uint256 registeredSamples = _registerAvailableMasks(
            roundId,
            numberOfPlayers,
            totalCoalitions,
            modelHash
        );
        console.log("  registeredSamples", registeredSamples);

        vm.stopPrank();

        bytes memory callData = abi.encodeCall(
            ContributionCalculator.calculateContribution,
            (roundId, 1, numberOfPlayers)
        );
        uint256 gasBefore = gasleft();
        (bool success, bytes memory returnData) = address(calculator)
            .staticcall{gas: DEBUG_GAS_LIMIT}(callData);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("  gasLimit", DEBUG_GAS_LIMIT);
        console.log("  gasUsedByCallFrame", gasUsed);
        console.log("  callSuccess", success);

        if (success && returnData.length >= 32) {
            int256 contribution = abi.decode(returnData, (int256));
            console.logInt(contribution);
        } else {
            console.log("  returnDataLength", returnData.length);
        }
    }
}

contract ContributionCalculatorFullCoalitionsDebugTest is Test {
    ContributionCalculator public implementation;
    ContributionCalculator public calculator;
    address public admin;

    uint8 internal constant DEBUG_NUM_TRAINERS = 16;
    uint256 internal constant DEBUG_ROUND_ID = 999;
    uint256 internal constant DEBUG_GAS_LIMIT = 60_000_000;

    function _debugResultForMask(uint256 mask) internal pure returns (int256) {
        return int256(mask * 100);
    }

    function _registerAvailableMasks(
        uint256 roundId,
        uint8 numberOfPlayers,
        uint256 requestedSamples,
        bytes32 modelHash
    ) internal returns (uint256 registeredSamples) {
        calculator.setEvaluationsRequired(roundId, requestedSamples);

        for (uint256 sampleId = 0; sampleId < requestedSamples; ++sampleId) {
            try calculator.getMask(roundId, sampleId, numberOfPlayers) returns (
                uint256 setId
            ) {
                calculator.registerResult(
                    roundId,
                    sampleId,
                    setId,
                    modelHash,
                    _debugResultForMask(setId),
                    numberOfPlayers
                );
                registeredSamples += 1;
            } catch {
                console.log("  samplerStoppedAt", sampleId);
                break;
            }
        }

        calculator.setEvaluationsRequired(roundId, registeredSamples);
    }

    function setUp() public {
        admin = makeAddr("admin");

        implementation = new ContributionCalculator();

        bytes memory initData = abi.encodeWithSelector(
            ContributionCalculator.initialize.selector,
            admin,
            2
        );

        ERC1967Proxy proxy = new ERC1967Proxy(
            address(implementation),
            initData
        );
        calculator = ContributionCalculator(address(proxy));

        uint8 numberOfPlayers = DEBUG_NUM_TRAINERS;
        uint256 roundId = DEBUG_ROUND_ID;
        uint256 totalCoalitions = calculator.getTotalEvaluations(
            roundId,
            numberOfPlayers
        );
        bytes32 modelHash = keccak256("debug_shapley_model");

        console.log("debug full coalitions setup");
        console.log("  numberOfPlayers", uint256(numberOfPlayers));
        console.log("  requestedSamples", totalCoalitions);

        vm.startPrank(admin);
        uint256 registeredSamples = _registerAvailableMasks(
            roundId,
            numberOfPlayers,
            totalCoalitions,
            modelHash
        );
        console.log("  registeredSamples", registeredSamples);

        vm.stopPrank();
    }

    /// @notice Change DEBUG_NUM_TRAINERS and rerun this single test.
    ///         Registration happens in setUp() so the measured call can use its own 60M gas frame.
    function test_calculateShapleyValue_fullCoalitions_debug_60m() public {
        uint8 numberOfPlayers = DEBUG_NUM_TRAINERS;
        console.log("numberOfPlayers", numberOfPlayers);
        uint256 roundId = DEBUG_ROUND_ID;

        bytes memory callData = abi.encodeCall(
            ContributionCalculator.calculateContribution,
            (roundId, 0, numberOfPlayers)
        );
        uint256 gasBefore = gasleft();
        (bool success, bytes memory returnData) = address(calculator)
            .staticcall{gas: DEBUG_GAS_LIMIT}(callData);
        uint256 gasUsed = gasBefore - gasleft();

        console.log("debug full coalitions call");
        console.log("  gasLimit", DEBUG_GAS_LIMIT);
        console.log("  gasUsedByCallFrame", gasUsed);
        console.log("  callSuccess", success);

        if (success && returnData.length >= 32) {
            int256 contribution = abi.decode(returnData, (int256));
            console.logInt(contribution);
        } else {
            console.log("  returnDataLength", returnData.length);
        }
    }
}
