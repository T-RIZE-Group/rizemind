// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.0;

import "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

import {SimpleMintCompensation} from "@rizemind-contracts/compensation/SimpleMintCompensation.sol";

contract SimpleMintCompensationTest is Test {
    SimpleMintCompensation public implementation;
    ERC1967Proxy public proxy;
    SimpleMintCompensation public token;
    address public aggregator;
    address public minter;
    address[] public trainers;
    address public nonTrainer;

    // max_rewards = 3 * 10**18
    uint256 constant maxRewards = 3 * 10 ** 18;

    function setUp() public {
        // Define our test accounts.
        aggregator = vm.addr(1); // equivalent to accounts[0] in ApeWorx
        minter = vm.addr(2); // minter role
        trainers.push(vm.addr(3)); // accounts[1]
        trainers.push(vm.addr(4)); // accounts[2]
        trainers.push(vm.addr(5)); // accounts[3]
        nonTrainer = vm.addr(6); // accounts[4]

        // Deploy implementation contract
        implementation = new SimpleMintCompensation();
        
        // Deploy proxy with initialization data
        bytes memory initData = abi.encodeWithSelector(
            SimpleMintCompensation.initialize.selector,
            "Test",
            "tst",
            maxRewards,
            aggregator,
            minter
        );
        proxy = new ERC1967Proxy(address(implementation), initData);
        token = SimpleMintCompensation(address(proxy));
    }

    function testDistribute() public {
        uint256 roundId = 1;
        uint64[] memory rewards = new uint64[](3);
        rewards[0] = uint64(10 ** 6);
        rewards[1] = uint64(2 * 10 ** 3);
        rewards[2] = uint64(50);

        vm.prank(minter);
        token.distribute(roundId, trainers, rewards);

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

    function testDistributeOnlyMinter() public {
        uint256 roundId = 1;
        uint64[] memory rewards = new uint64[](3);
        rewards[0] = uint64(10 ** 6);
        rewards[1] = uint64(2 * 10 ** 3);
        rewards[2] = uint64(50);

        // Test that non-minter cannot distribute
        vm.prank(nonTrainer);
        vm.expectRevert();
        token.distribute(roundId, trainers, rewards);

        // Test that aggregator (admin) cannot distribute without minter role
        vm.prank(aggregator);
        vm.expectRevert();
        token.distribute(roundId, trainers, rewards);

        // Test that minter can distribute
        vm.prank(minter);
        token.distribute(roundId, trainers, rewards);
    }

    function testInitializeCanOnlyBeCalledOnce() public {
        // Test that initialize cannot be called twice
        vm.expectRevert();
        token.initialize("Test2", "tst2", maxRewards, aggregator, minter);
    }

    function testTokenMetadata() public view {
        assertEq(token.name(), "Test", "Token name should be set correctly");
        assertEq(token.symbol(), "tst", "Token symbol should be set correctly");
        assertEq(token.decimals(), 18, "Token decimals should be 18");
    }

    function testRoleAssignment() public view {
        // Test that roles are assigned correctly
        assertTrue(token.hasRole(token.DEFAULT_ADMIN_ROLE(), aggregator), "Aggregator should have admin role");
        assertTrue(token.hasRole(token.MINTER_ROLE(), minter), "Minter should have minter role");
        assertFalse(token.hasRole(token.MINTER_ROLE(), aggregator), "Aggregator should not have minter role");
    }
}
