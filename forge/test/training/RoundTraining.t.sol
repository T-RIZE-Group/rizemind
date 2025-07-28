// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.10;

import "forge-std/Test.sol";
import {RoundTraining} from "@rizemind-contracts/training/RoundTraining.sol";
import {RoundSummary, RoundFinished} from "@rizemind-contracts/training/ITraining.sol";

contract InitializableRoundTraining is RoundTraining {
    function initialize() public initializer {
        __RoundTraining_init();
    }
}

contract RoundTrainingTest is Test {
    InitializableRoundTraining public roundTraining;

    function setUp() public {
        roundTraining = new InitializableRoundTraining();
        roundTraining.initialize();
    }

    function testInitialRound() public view {
        assertEq(roundTraining.currentRound(), 1, "Initial round should be 1");
    }

    function testNextRoundSuccess() public {
        RoundSummary memory summary = RoundSummary({
            roundId: 1,
            nTrainers: 3,
            modelScore: 100,
            totalContributions: 1000
        });
        vm.expectEmit(true, false, false, true);
        emit RoundFinished(1, 3, 100, 1000);
        roundTraining.nextRound(summary);
        assertEq(roundTraining.currentRound(), 2, "Round should increment");
    }

    function testNextRoundRevertsOnMismatch() public {
        RoundSummary memory summary = RoundSummary({
            roundId: 2,
            nTrainers: 3,
            modelScore: 100,
            totalContributions: 1000
        });
        vm.expectRevert();
        roundTraining.nextRound(summary);
    }

    function testSupportsInterface() public view {
        assertTrue(
            roundTraining.supportsInterface(
                roundTraining.currentRound.selector
            ),
            "Should support currentRound"
        );
        assertTrue(
            roundTraining.supportsInterface(roundTraining.nextRound.selector),
            "Should support nextRound"
        );
        assertFalse(
            roundTraining.supportsInterface(0xdeadbeef),
            "Should not support random selector"
        );
    }
}
