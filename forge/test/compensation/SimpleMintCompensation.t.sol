// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "forge-std/Test.sol";

import {SimpleMintCompensation} from "@rizemind-contracts/compensation/SimpleMintCompensation.sol";

contract TESTSimpleMintCompensation is SimpleMintCompensation {
    function initialize(
        string memory name,
        string memory symbol,
        uint256 maxRewards
    ) public initializer {
        __SimpleMintCompensation_init(name, symbol, maxRewards);
    }

    function distribute(
        address[] calldata trainers,
        uint64[] calldata contributions
    ) external {
        _distribute(trainers, contributions);
    }
}

contract TESTSimpleMintCompensationTest is Test {
    TESTSimpleMintCompensation public token;
    address public aggregator;
    address[] public trainers;
    address public nonTrainer;

    // max_rewards = 3 * 10**18
    uint256 constant maxRewards = 3 * 10 ** 18;

    function setUp() public {
        // Define our test accounts.
        aggregator = vm.addr(1); // equivalent to accounts[0] in ApeWorx
        trainers.push(vm.addr(2)); // accounts[1]
        trainers.push(vm.addr(3)); // accounts[2]
        trainers.push(vm.addr(4)); // accounts[3]
        nonTrainer = vm.addr(5); // accounts[4]

        // Deploy and initialize the token contract as aggregator.
        vm.prank(aggregator);
        token = new TESTSimpleMintCompensation();
        vm.prank(aggregator);
        token.initialize("Test", "tst", maxRewards);
    }

    function testDistribute() public {
        uint64[] memory rewards = new uint64[](3);
        rewards[0] = uint64(10 ** 6);
        rewards[1] = uint64(2 * 10 ** 3);
        rewards[2] = uint64(50);

        vm.prank(aggregator);
        token.distribute(trainers, rewards);

        uint256 expectedTrainer0 = maxRewards;
        uint256 expectedTrainer1 = (2 * maxRewards) / 10 ** 3;
        uint256 expectedTrainer2 = (50 * maxRewards) / 10 ** 6;

        // Check the balances.
        assertEq(
            token.balanceOf(trainers[0]),
            expectedTrainer0,
            "should have received max rewards"
        );
        assertEq(
            token.balanceOf(trainers[1]),
            expectedTrainer1,
            "incorrect reward for trainer[1]"
        );
        assertEq(
            token.balanceOf(trainers[2]),
            expectedTrainer2,
            "incorrect reward for trainer[2]"
        );
    }
}
