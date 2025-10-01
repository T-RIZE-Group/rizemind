// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.10;

import "forge-std/Test.sol";
import {RoundTraining} from "@rizemind-contracts/training/RoundTraining.sol";
import {RoundSummary, RoundFinished} from "@rizemind-contracts/training/ITraining.sol";

contract InitializableRoundTraining is RoundTraining {
    function initialize() public initializer {
        __RoundTraining_init();
    }

    function nextRound() external {
        _nextRound();
    }
}

contract RoundTrainingTest is Test {
    InitializableRoundTraining public roundTraining;

    function setUp() public {
        roundTraining = new InitializableRoundTraining();
        roundTraining.initialize();
    }

    function testInitialRound() public view {
        assertEq(roundTraining.currentRound(), 0, "Initial round should be 0");
    }

    function testNextRoundSuccess() public {
        uint256 round = roundTraining.currentRound();
        roundTraining.nextRound();
        assertEq(roundTraining.currentRound(), round + 1, "Round should increment");
    }

    function testSupportsInterface() public view {
        assertTrue(
            roundTraining.supportsInterface(
                roundTraining.currentRound.selector
            ),
            "Should support currentRound"
        );
        assertFalse(
            roundTraining.supportsInterface(0xdeadbeef),
            "Should not support random selector"
        );
    }
}
