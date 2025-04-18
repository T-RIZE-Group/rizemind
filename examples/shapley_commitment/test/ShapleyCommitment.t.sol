// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../contracts/ShapleyCommitment.sol";

contract ShapleyCommitmentTest is Test {
    ShapleyCommitment public shapleyCommitment;
    address coordinator;
    address tester;

    function setUp() public {
        shapleyCommitment = new ShapleyCommitment();
        coordinator = address(this);
        tester = vm.addr(2);
    }

    function testPublishCoalitionRoot() public {
        bytes32 coalitionId = keccak256("testCoalition");
        bytes32 coalitionRoot = keccak256("testRoot");

        vm.expectEmit(true, false, false, true);
        emit ShapleyCommitment.CoalitionRootPublished(coalitionId, coalitionRoot);

        shapleyCommitment.publishCoalitionRoot(coalitionId, coalitionRoot);

        // Verify root was stored
        assertEq(shapleyCommitment.coalitionRoots(coalitionId), coalitionRoot);
    }

    function testCannotPublishDuplicateRoot() public {
        bytes32 coalitionId = keccak256("testCoalition");
        bytes32 coalitionRoot = keccak256("testRoot");

        shapleyCommitment.publishCoalitionRoot(coalitionId, coalitionRoot);

        // Attempt to publish same root again should fail
        vm.expectRevert("Root already exists");
        shapleyCommitment.publishCoalitionRoot(coalitionId, coalitionRoot);
    }

    function testPublishResult() public {
        bytes32 coalitionId = keccak256("testCoalition");
        bytes32 coalitionRoot = keccak256("testRoot");
        uint256 result = 100;

        // First publish the root
        shapleyCommitment.publishCoalitionRoot(coalitionId, coalitionRoot);

        // Now publish result as tester
        vm.prank(tester);
        vm.expectEmit(true, false, false, true);
        emit ShapleyCommitment.CoalitionResultPublished(coalitionId, result);

        shapleyCommitment.publishResult(coalitionId, result);

        // Verify result was stored
        assertEq(shapleyCommitment.coalitionResults(coalitionId), result);
    }

    function testCannotPublishResultWithoutRoot() public {
        bytes32 coalitionId = keccak256("testCoalition");
        uint256 result = 100;

        // Attempt to publish result without publishing root first
        vm.expectRevert("Coalition root not published");
        shapleyCommitment.publishResult(coalitionId, result);
    }

    function testCannotPublishDuplicateResult() public {
        bytes32 coalitionId = keccak256("testCoalition");
        bytes32 coalitionRoot = keccak256("testRoot");
        uint256 result = 100;

        // First publish the root
        shapleyCommitment.publishCoalitionRoot(coalitionId, coalitionRoot);

        // Publish result first time
        vm.prank(tester);
        shapleyCommitment.publishResult(coalitionId, result);

        // Attempt to publish same result again should fail
        vm.expectRevert("Result already published");
        vm.prank(tester);
        shapleyCommitment.publishResult(coalitionId, result);
    }
}