// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {ShapleyValueCalculator} from "../../src/contribution/ShapleyValueCalculator.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

contract MockShapleyValueCalculator is ShapleyValueCalculator {
    function initialize() external initializer {
        __EvaluationStorage_init();
    }

    function exposed_getTotalEvaluations(uint256 roundId, uint8 numberOfPlayers) external view returns (uint256) {
        return _getTotalEvaluations(roundId, numberOfPlayers);
    }

    function exposed_getEvaluationsRequired(uint256 roundId, uint8 numberOfPlayers) external view returns (uint256) {
        return _getEvaluationsRequired(roundId, numberOfPlayers);
    }

    function exposed_registerResult(
        uint256 roundId,
        uint256 sampleId,
        uint256 setId,
        bytes32 modelHash,
        int256 result,
        uint8 numberOfPlayers
    ) external {
        _registerResult(roundId, sampleId, setId, modelHash, result, numberOfPlayers);
    }

    function exposed_calcShapley(uint256 roundId, uint256 trainerIndex, uint8 numberOfPlayers)
        external
        view
        returns (int256)
    {
        return _calcShapley(roundId, trainerIndex, numberOfPlayers);
    }

    function exposed_weight(uint256 n, uint256 s) external view returns (uint256) {
        return weight(n, s);
    }

    function exposed_popcount(uint256 x) external pure returns (uint256) {
        return popcount(x);
    }

    function exposed_setNumSamples(uint256 roundId, uint256 numSamples) external {
        _setNumSamples(roundId, numSamples);
    }

    function exposed_getMaskMonteCarlo(uint256 roundId, uint256 i, uint8 numberOfPlayers)
        external
        view
        returns (uint256)
    {
        return _get_mask_monte_carlo(roundId, i, numberOfPlayers);
    }

    function exposed_getMaskStratified(uint256 roundId, uint256 i, uint8 numberOfPlayers)
        external
        view
        returns (uint256)
    {
        return _get_mask_stratified(roundId, i, numberOfPlayers);
    }

    function exposed_getMaskStratifiedMonteCarlo(uint256 roundId, uint256 i, uint8 numberOfPlayers)
        external
        view
        returns (uint256)
    {
        return _get_mask_stratified_monte_carlo(roundId, i, numberOfPlayers);
    }

    function _getMask(uint256, uint256 i, uint8 numberOfPlayers) internal view override returns (uint256) {
        return i % (1 << numberOfPlayers);
    }
}

contract ShapleyValueCalculatorTest is Test {
    MockShapleyValueCalculator public implementation;
    MockShapleyValueCalculator public calculator;
    address public admin;
    address public user;

    function setUp() public {
        admin = makeAddr("admin");
        user = makeAddr("user");

        implementation = new MockShapleyValueCalculator();

        bytes memory initData = abi.encodeWithSelector(MockShapleyValueCalculator.initialize.selector);

        ERC1967Proxy proxy = new ERC1967Proxy(address(implementation), initData);
        calculator = MockShapleyValueCalculator(address(proxy));
    }

    function test_initialize() public view {
        assertTrue(address(calculator) != address(0));
    }

    function test_basicShapleyCalculation() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 2;
        uint256 numSamples = 4;

        calculator.exposed_setNumSamples(roundId, numSamples);

        assertEq(calculator.exposed_getEvaluationsRequired(roundId, numberOfPlayers), numSamples);

        uint256 totalEvaluations = calculator.exposed_getTotalEvaluations(roundId, numberOfPlayers);
        assertEq(totalEvaluations, 1 << numberOfPlayers);

        bytes32 modelHash = keccak256("test_model");

        uint256 targetSetId0 = calculator.getMask(roundId, 0, numberOfPlayers);
        uint256 targetSetId1 = calculator.getMask(roundId, 1, numberOfPlayers);
        uint256 targetSetId2 = calculator.getMask(roundId, 2, numberOfPlayers);
        uint256 targetSetId3 = calculator.getMask(roundId, 3, numberOfPlayers);

        calculator.exposed_registerResult(roundId, 0, targetSetId0, modelHash, 0, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 1, targetSetId1, modelHash, 300, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 2, targetSetId2, modelHash, 600, numberOfPlayers);
        calculator.exposed_registerResult(roundId, 3, targetSetId3, modelHash, 1500, numberOfPlayers);

        int256 shapleyValue0 = calculator.exposed_calcShapley(roundId, 0, numberOfPlayers);
        int256 shapleyValue1 = calculator.exposed_calcShapley(roundId, 1, numberOfPlayers);

        assertEq(shapleyValue0, 600);
        assertEq(shapleyValue1, 900);
    }

    function test_registerResult_withValidDistance() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 2;
        bytes32 modelHash = keccak256("test_model");

        uint256 targetSetId = calculator.getMask(roundId, 0, numberOfPlayers);

        calculator.exposed_registerResult(roundId, 0, targetSetId, modelHash, 100, numberOfPlayers);

        uint256 nearbySetId = targetSetId ^ 1;
        calculator.exposed_registerResult(roundId, 0, nearbySetId, modelHash, 150, numberOfPlayers);

        assertEq(calculator.getResult(roundId, targetSetId), 100);
        assertEq(calculator.getResult(roundId, nearbySetId), 150);
    }

    function test_registerResult_withInvalidDistance() public {
        uint256 roundId = 1;
        uint8 numberOfPlayers = 5;
        bytes32 modelHash = keccak256("test_model");

        uint256 targetSetId = calculator.getMask(roundId, 0, numberOfPlayers);
        uint256 farSetId = targetSetId ^ 3;

        vm.expectRevert(
            abi.encodeWithSelector(ShapleyValueCalculator.setIdTooFar.selector, roundId, farSetId, targetSetId)
        );

        calculator.exposed_registerResult(roundId, 0, farSetId, modelHash, 200, numberOfPlayers);
    }

    function test_getMaskMonteCarlo_returnsValuesInRange() public view {
        uint256 roundId = 7;
        uint8 numberOfPlayers = 5;
        uint256 totalMasks = 1 << numberOfPlayers;

        for (uint256 i = 0; i < 64; ++i) {
            uint256 mask = calculator.exposed_getMaskMonteCarlo(roundId, i, numberOfPlayers);
            assertLt(mask, totalMasks);
        }
    }

    function test_getMaskMonteCarlo_allowsRepeatedDrawsBeyondDomainSize() public view {
        uint256 roundId = 8;
        uint8 numberOfPlayers = 2;
        uint256 totalMasks = 1 << numberOfPlayers;
        uint256[] memory counts = new uint256[](totalMasks);
        bool foundDuplicate = false;

        for (uint256 i = 0; i < 8; ++i) {
            uint256 mask = calculator.exposed_getMaskMonteCarlo(roundId, i, numberOfPlayers);
            counts[mask] += 1;
            if (counts[mask] > 1) {
                foundDuplicate = true;
            }
        }

        assertTrue(foundDuplicate);
    }

    function test_getMaskStratified_returnsUniqueNonFullCoalitions() public {
        uint256 roundId = 9;
        uint8 numberOfPlayers = 5;
        uint256 emittedBudget = 20;
        uint256 totalMasks = 1 << numberOfPlayers;
        bool[] memory seen = new bool[](totalMasks);

        calculator.exposed_setNumSamples(roundId, emittedBudget);

        for (uint256 i = 0; i < emittedBudget; ++i) {
            uint256 mask = calculator.exposed_getMaskStratified(roundId, i, numberOfPlayers);
            assertLt(mask, totalMasks);
            assertLt(calculator.exposed_popcount(mask), numberOfPlayers);
            assertFalse(seen[mask]);
            seen[mask] = true;
        }
    }

    function test_getMaskStratified_coversAllNonFullCoalitions_whenBudgetSaturates() public {
        uint256 roundId = 10;
        uint8 numberOfPlayers = 3;
        uint256 totalMasks = 1 << numberOfPlayers;
        bool[] memory seen = new bool[](totalMasks);

        calculator.exposed_setNumSamples(roundId, 100);

        for (uint256 i = 0; i < totalMasks - 1; ++i) {
            uint256 mask = calculator.exposed_getMaskStratified(roundId, i, numberOfPlayers);
            assertFalse(seen[mask]);
            seen[mask] = true;
        }

        for (uint256 mask = 0; mask < totalMasks - 1; ++mask) {
            assertTrue(seen[mask]);
        }
        assertFalse(seen[totalMasks - 1]);
    }

    function test_getMaskStratified_revertsBeyondEmittedBudget() public {
        uint256 roundId = 11;
        uint8 numberOfPlayers = 3;

        calculator.exposed_setNumSamples(roundId, 100);

        vm.expectRevert(bytes("stratified sampleId out of range"));
        calculator.exposed_getMaskStratified(roundId, 7, numberOfPlayers);
    }

    function test_getMaskStratifiedMonteCarlo_returnsValuesInRange() public {
        uint256 roundId = 12;
        uint8 numberOfPlayers = 5;
        uint256 totalMasks = 1 << numberOfPlayers;

        calculator.exposed_setNumSamples(roundId, 64);

        for (uint256 i = 0; i < 64; ++i) {
            uint256 mask = calculator.exposed_getMaskStratifiedMonteCarlo(roundId, i, numberOfPlayers);
            assertLt(mask, totalMasks);
            assertLt(calculator.exposed_popcount(mask), numberOfPlayers);
        }
    }

    function test_getMaskStratifiedMonteCarlo_allowsDuplicates() public {
        uint256 roundId = 13;
        uint8 numberOfPlayers = 2;
        uint256 totalMasks = 1 << numberOfPlayers;
        uint256[] memory counts = new uint256[](totalMasks);
        bool foundDuplicate = false;

        calculator.exposed_setNumSamples(roundId, 8);

        for (uint256 i = 0; i < 8; ++i) {
            uint256 mask = calculator.exposed_getMaskStratifiedMonteCarlo(roundId, i, numberOfPlayers);
            counts[mask] += 1;
            if (counts[mask] > 1) {
                foundDuplicate = true;
            }
        }

        assertTrue(foundDuplicate);
    }
}
