// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {SelectorFactory} from "../../src/sampling/SelectorFactory.sol";
import {AlwaysSampled} from "../../src/sampling/AlwaysSampled.sol";
import {ISelector} from "../../src/sampling/ISelector.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";

/// @title MockInvalidSelector
/// @notice Mock contract that doesn't implement ISelector interface
contract MockInvalidSelector {
    // This contract intentionally doesn't implement ISelector
}

contract AlwaysSampledV2Mock is AlwaysSampled {
        /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        override(AlwaysSampled)
        returns (string memory)
    {
        return "always-sampled-v2.0.0";
    }
}

/// @title MockSelectorWithInit
/// @notice Mock selector that requires initialization parameters
contract MockSelectorWithInit is AlwaysSampled {
    uint256 public targetRatio;
    address public owner;
    bool public initialized;

    event Initialized(uint256 targetRatio, address owner);

    function initialize(uint256 _targetRatio, address _owner) external {
        require(!initialized, "Already initialized");
        targetRatio = _targetRatio;
        owner = _owner;
        initialized = true;
        emit Initialized(_targetRatio, _owner);
    }

    function isSelected(address, uint256) external pure override returns (bool) {
        return true;
    }

    function supportsInterface(bytes4 interfaceId) external pure override returns (bool) {
        return interfaceId == type(ISelector).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }

            /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        override(AlwaysSampled)
        returns (string memory)
    {
        return "mock-selector-with-init-v1.0.0";
    }

    function getTargetRatio() external view returns (uint256) {
        return targetRatio;
    }

    function getOwner() external view returns (address) {
        return owner;
    }
}

contract SelectorFactoryTest is Test {
    SelectorFactory public factory;
    AlwaysSampled public alwaysSampled;
    MockSelectorWithInit public mockSelectorWithInit;

    address public owner;
    address public user;

    bytes32 public alwaysSampledId;
    bytes32 public mockSelectorWithInitId;

    event SelectorImplementationRegistered(bytes32 indexed id, address indexed implementation);
    event SelectorInstanceCreated(bytes32 indexed id, address indexed instance, address indexed creator);
    event SelectorImplementationUpdated(bytes32 indexed id, address indexed oldImplementation, address indexed newImplementation);
    event SelectorImplementationRemoved(bytes32 indexed id, address indexed implementation);

    function setUp() public {
        owner = makeAddr("owner");
        user = makeAddr("user");

        vm.startPrank(owner);
        // Deploy contracts
        factory = new SelectorFactory(owner);
        alwaysSampled = new AlwaysSampled();
        factory.registerSelectorImplementation(address(alwaysSampled));
        (,, string memory version,,,,) = alwaysSampled.eip712Domain();
        alwaysSampledId = factory.getID(version);
        mockSelectorWithInit = new MockSelectorWithInit();
        factory.registerSelectorImplementation(address(mockSelectorWithInit));
        (,, string memory version2,,,,) = mockSelectorWithInit.eip712Domain();
        mockSelectorWithInitId = factory.getID(version2);
        vm.stopPrank();
    }

    // ============================================================================
    // REGISTRATION TESTS
    // ============================================================================

    function test_registerSelectorImplementation_AlwaysSampled_success() public {
        vm.startPrank(owner);
        AlwaysSampledV2Mock alwaysSampledV2Mock = new AlwaysSampledV2Mock();
        vm.expectEmit(true, true, false, false);
        bytes32 actualId = factory.getID("always-sampled-v2.0.0");
        emit SelectorImplementationRegistered(actualId, address(alwaysSampledV2Mock));

        factory.registerSelectorImplementation(address(alwaysSampledV2Mock));

        assertTrue(factory.isSelectorRegistered(actualId), "Selector should be registered");
        assertEq(factory.getSelectorImplementation(actualId), address(alwaysSampledV2Mock), "Implementation should be set");
    }

    function test_registerSelectorImplementation_onlyOwner() public {
        vm.startPrank(user);

        vm.expectRevert();
        factory.registerSelectorImplementation(address(alwaysSampled));
    }

    function test_registerSelectorImplementation_alreadyExists() public {
        vm.startPrank(owner);

        vm.expectRevert(SelectorFactory.SelectorImplementationAlreadyExists.selector);
        factory.registerSelectorImplementation(address(alwaysSampled));
    }

    function test_registerSelectorImplementation_zeroAddress() public {
        vm.startPrank(owner);

        vm.expectRevert(SelectorFactory.SelectorImplementationInvalid.selector);
        factory.registerSelectorImplementation(address(0));
    }

    function test_registerSelectorImplementation_invalidInterface() public {
        vm.startPrank(owner);
        MockInvalidSelector invalidSelector = new MockInvalidSelector();
        vm.expectRevert(SelectorFactory.SelectorImplementationInvalid.selector);
        factory.registerSelectorImplementation(address(invalidSelector));
    }

    // ============================================================================
    // CREATION TESTS
    // ============================================================================

    function test_createSelector_AlwaysSampled_success() public {

        vm.startPrank(user);

        bytes32 salt = keccak256(abi.encodePacked("test_salt"));
        bytes memory constructorData = ""; // No constructor data needed for AlwaysSampled

        // Don't expect specific address in event since it's generated
        vm.expectEmit(true, false, false, false);
        emit SelectorInstanceCreated(alwaysSampledId, address(0), user); // address(0) placeholder

        address instance = factory.createSelector(alwaysSampledId, salt, constructorData);

        assertTrue(instance != address(0), "Instance should be created");
        
        // Verify the instance works correctly
        ISelector selectorInstance = ISelector(instance);
        assertTrue(selectorInstance.isSelected(user, 1), "Instance should work correctly");
        assertTrue(selectorInstance.isSelected(user, 2), "Instance should work correctly");
        assertTrue(selectorInstance.isSelected(user, 100), "Instance should work correctly");
    }

    function test_createSelector_withInitData() public {

        // Encode init function call: targetRatio = 80%, owner = user
        bytes memory initData = abi.encodeWithSelector(
            MockSelectorWithInit.initialize.selector,
            80 * 10**18,
            user
        );

        bytes32 salt = keccak256(abi.encodePacked("test_salt_with_init"));

        vm.startPrank(user);
        address instance = factory.createSelector(mockSelectorWithInitId, salt, initData);

        assertTrue(instance != address(0), "Instance should be created");
        
        // Verify the instance was initialized correctly via init function
        MockSelectorWithInit selectorInstance = MockSelectorWithInit(instance);
        assertTrue(selectorInstance.isSelected(user, 1), "Instance should work correctly");
        assertEq(selectorInstance.getTargetRatio(), 80 * 10**18, "Target ratio should be set");
        assertEq(selectorInstance.getOwner(), user, "Owner should be set");
        assertTrue(selectorInstance.initialized(), "Instance should be marked as initialized");
    }

    function test_createSelector_notFound() public {
        vm.startPrank(user);

        bytes32 salt = keccak256(abi.encodePacked("test_salt"));
        vm.expectRevert(SelectorFactory.SelectorImplementationNotFound.selector);
        factory.createSelector(salt, salt, "");
    }

    function test_createSelector_initFailure() public {

        // Test with invalid init data that will cause the call to fail
        bytes memory invalidInitData = abi.encodeWithSelector(
            0x12345678, // Invalid function selector
            0x12345678  // Invalid data
        );

        bytes32 salt = keccak256(abi.encodePacked("test_salt_failure"));

        vm.startPrank(user);
        vm.expectRevert(); // Expect any revert since the error type may vary
        factory.createSelector(mockSelectorWithInitId, salt, invalidInitData);
    }

    function test_createSelector_deterministicAddresses() public {

        bytes32 salt1 = keccak256(abi.encodePacked("salt1"));
        bytes32 salt2 = keccak256(abi.encodePacked("salt2"));

        vm.startPrank(user);

        address instance1 = factory.createSelector(alwaysSampledId, salt1, "");
        address instance2 = factory.createSelector(alwaysSampledId, salt2, "");

        assertTrue(instance1 != address(0), "First instance should be created");
        assertTrue(instance2 != address(0), "Second instance should be created");

        // Different salts should produce different addresses
        assertTrue(instance1 != instance2, "Different salts should produce different addresses");
        
        // Note: CREATE2 deterministic addresses not yet implemented in SelectorFactory
        // This test will be updated when CREATE2 support is added
    }

    function test_createSelector_multipleUsers() public {
        address user1 = makeAddr("user1");
        address user2 = makeAddr("user2");
        address user3 = makeAddr("user3");

        bytes32 salt1 = keccak256(abi.encodePacked("user1_salt"));
        bytes32 salt2 = keccak256(abi.encodePacked("user2_salt"));
        bytes32 salt3 = keccak256(abi.encodePacked("user3_salt"));

        // User 1 creates instance
        vm.prank(user1);
        address instance1 = factory.createSelector(alwaysSampledId, salt1, "");

        // User 2 creates instance
        vm.prank(user2);
        address instance2 = factory.createSelector(alwaysSampledId, salt2, "");

        // User 3 creates instance
        vm.prank(user3);
        address instance3 = factory.createSelector(alwaysSampledId, salt3, "");

        // All instances should be different
        assertTrue(instance1 != instance2, "Instances should be different");
        assertTrue(instance1 != instance3, "Instances should be different");
        assertTrue(instance2 != instance3, "Instances should be different");

        // All instances should work
        ISelector selector1 = ISelector(instance1);
        ISelector selector2 = ISelector(instance2);
        ISelector selector3 = ISelector(instance3);

        assertTrue(selector1.isSelected(user1, 1), "Instance 1 should work");
        assertTrue(selector2.isSelected(user2, 1), "Instance 2 should work");
        assertTrue(selector3.isSelected(user3, 1), "Instance 3 should work");
    }

    // ============================================================================
    // VIEW FUNCTION TESTS
    // ============================================================================

    function test_getSelectorImplementation_success() public {

        address implementation = factory.getSelectorImplementation(alwaysSampledId);
        assertEq(implementation, address(alwaysSampled), "Should return correct implementation");
    }

    function test_getSelectorImplementation_notRegistered() public view {
        address implementation = factory.getSelectorImplementation(hex"0122");
        assertEq(implementation, address(0), "Should return zero address for unregistered selector");
    }

    function test_isSelectorRegistered() public {
        assertFalse(factory.isSelectorVersionRegistered("0122"), "Should not be registered initially");

        assertTrue(factory.isSelectorRegistered(alwaysSampledId), "Should be registered after registration");
    }

}
