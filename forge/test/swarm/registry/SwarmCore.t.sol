// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Test} from "forge-std/Test.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";
import {SwarmCore} from "../../../src/swarm/registry/SwarmCore.sol";
import {AlwaysSampled} from "../../../src/sampling/AlwaysSampled.sol";
import {ISelector} from "../../../src/sampling/ISelector.sol";
import {IContributionCalculator} from "../../../src/contribution/types.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";

/// @title MockEvaluatorSelector
/// @notice Mock contract that implements IEvaluatorSelector interface
contract MockSelector is AlwaysSampled {

}

/// @title MockSwarmCore
/// @notice Mock contract that exposes internal functions for testing
contract MockSwarmCore is SwarmCore {
    /// @notice Expose the internal _updateTrainerSelector function for testing
    function updateTrainerSelector(address newTrainerSelector) external {
        _updateTrainerSelector(newTrainerSelector);
    }

    /// @notice Expose the internal _updateEvaluatorSelector function for testing
    function updateEvaluatorSelector(address newEvaluatorSelector) external {
        _updateEvaluatorSelector(newEvaluatorSelector);
    }

    /// @notice Expose the internal _updateContributionCalculator function for testing
    function updateContributionCalculator(address newContributionCalculator) external {
        _updateContributionCalculator(newContributionCalculator);
    }
}


/// @title MockInvalidSelector
/// @notice Mock contract that doesn't implement ISelector interface
contract MockInvalidSelector {
    // This contract intentionally doesn't implement ISelector
}

/// @title MockContributionCalculator
/// @notice Mock contract that implements IContributionCalculator interface
contract MockContributionCalculator {
    function calculateContribution(uint256 roundId, uint256 trainerIndex, uint8 numberOfTrainers) external view returns (int256) {
        return int256(1);
    }
    
    function getResult(uint256 roundId, uint256 setId) external view returns (int256) {
        return int256(1);
    }
    
    function getResultOrThrow(uint256 roundId, uint256 setId) external view returns (int256) {
        return int256(1);
    }
    
    function eip712Domain() external view returns (bytes1 fields, string memory name, string memory version, uint256 chainId, address verifyingContract, bytes32 salt, uint256[] memory extensions) {
        return (0x0, "MockCalculator", "1", block.chainid, address(this), bytes32(0), new uint256[](0));
    }
    
    function supportsInterface(bytes4 interfaceId) external view returns (bool) {
        return true;
    }
}

contract SwarmCoreTest is Test {
    MockSwarmCore public implementation;
    ERC1967Proxy public proxy;
    MockSwarmCore public swarmCore;
    AlwaysSampled public alwaysSampled;
    MockSelector public mockEvaluatorSelector;
    MockContributionCalculator public mockContributionCalculator;
    MockInvalidSelector public invalidEvaluatorSelector;

    event TrainerSelectorUpdated(address indexed previousSelector, address indexed newSelector);
    event EvaluatorSelectorUpdated(address indexed previousSelector, address indexed newSelector);
    event ContributionCalculatorUpdated(address indexed previousCalculator, address indexed newCalculator);

    function setUp() public {
        // Deploy contracts
        implementation = new MockSwarmCore();
        alwaysSampled = new AlwaysSampled();
        mockEvaluatorSelector = new MockSelector();
        mockContributionCalculator = new MockContributionCalculator();
        invalidEvaluatorSelector = new MockInvalidSelector();
        
        // Deploy proxy
        bytes memory initData = abi.encodeWithSelector(
            SwarmCore.initialize.selector,
            address(alwaysSampled),
            address(mockEvaluatorSelector),
            address(mockContributionCalculator)
        );
        proxy = new ERC1967Proxy(address(implementation), initData);
        swarmCore = MockSwarmCore(address(proxy));
    }

    // ============================================================================
    // BASIC FUNCTIONALITY TESTS
    // ============================================================================

    function test_initialize_setsTrainerSelector() public view {
        // Test that initialize sets the trainer selector correctly
        address trainerSelector = swarmCore.getTrainerSelector();
        assertEq(trainerSelector, address(alwaysSampled), "Trainer selector should be set to AlwaysSampled");
    }

    function test_initialize_setsEvaluatorSelector() public view {
        // Test that initialize sets the evaluator selector correctly
        address evaluatorSelector = swarmCore.getEvaluatorSelector();
        assertEq(evaluatorSelector, address(mockEvaluatorSelector), "Evaluator selector should be set to MockSelector");
    }

    function test_initialize_setsContributionCalculator() public view {
        // Test that initialize sets the contribution calculator correctly
        address contributionCalculator = swarmCore.getContributionCalculator();
        assertEq(contributionCalculator, address(mockContributionCalculator), "Contribution calculator should be set to MockContributionCalculator");
    }

    function test_initialize_canOnlyBeCalledOnce() public {
        // Test that initialize cannot be called twice
        vm.expectRevert();
        swarmCore.initialize(address(alwaysSampled), address(mockEvaluatorSelector), address(mockContributionCalculator));
    }

    function test_getTrainerSelector_returnsCorrectAddress() public view {
        // Test that getTrainerSelector returns the correct address
        address trainerSelector = swarmCore.getTrainerSelector();
        assertEq(trainerSelector, address(alwaysSampled), "Should return the correct trainer selector address");
    }

    function test_getEvaluatorSelector_returnsCorrectAddress() public view {
        // Test that getEvaluatorSelector returns the correct address
        address evaluatorSelector = swarmCore.getEvaluatorSelector();
        assertEq(evaluatorSelector, address(mockEvaluatorSelector), "Should return the correct evaluator selector address");
    }

    function test_getContributionCalculator_returnsCorrectAddress() public view {
        // Test that getContributionCalculator returns the correct address
        address contributionCalculator = swarmCore.getContributionCalculator();
        assertEq(contributionCalculator, address(mockContributionCalculator), "Should return the correct contribution calculator address");
    }

    function test_updateTrainerSelector_updatesStorage() public {
        // Test that _updateTrainerSelector updates the storage correctly
        address newTrainerSelector = address(new AlwaysSampled());

        vm.expectEmit(true, true, false, false);
        emit TrainerSelectorUpdated(address(alwaysSampled), newTrainerSelector);
        
        swarmCore.updateTrainerSelector(newTrainerSelector);
        
        address updatedSelector = swarmCore.getTrainerSelector();
        assertEq(updatedSelector, newTrainerSelector, "Trainer selector should be updated");
    }

    function test_updateEvaluatorSelector_updatesStorage() public {
        // Test that _updateEvaluatorSelector updates the storage correctly
        address newEvaluatorSelector = address(new MockSelector());

        vm.expectEmit(true, true, false, false);
        emit EvaluatorSelectorUpdated(address(mockEvaluatorSelector), newEvaluatorSelector);
        
        swarmCore.updateEvaluatorSelector(newEvaluatorSelector);
        
        address updatedSelector = swarmCore.getEvaluatorSelector();
        assertEq(updatedSelector, newEvaluatorSelector, "Evaluator selector should be updated");
    }

    function test_updateContributionCalculator_updatesStorage() public {
        // Test that _updateContributionCalculator updates the storage correctly
        address newContributionCalculator = address(new MockContributionCalculator());

        vm.expectEmit(true, true, false, false);
        emit ContributionCalculatorUpdated(address(mockContributionCalculator), newContributionCalculator);
        
        swarmCore.updateContributionCalculator(newContributionCalculator);
        
        address updatedCalculator = swarmCore.getContributionCalculator();
        assertEq(updatedCalculator, newContributionCalculator, "Contribution calculator should be updated");
    }

    function test_updateTrainerSelector_emitsEvent() public {
        // Test that _updateTrainerSelector emits the correct event
        address newTrainerSelector = address(new AlwaysSampled());
        
        vm.expectEmit(true, true, false, false);
        emit TrainerSelectorUpdated(address(alwaysSampled), newTrainerSelector);
        
        swarmCore.updateTrainerSelector(newTrainerSelector);
    }

    function test_updateEvaluatorSelector_emitsEvent() public {
        // Test that _updateEvaluatorSelector emits the correct event
        address newEvaluatorSelector = address(new MockSelector());

        vm.expectEmit(true, true, false, false);
        emit EvaluatorSelectorUpdated(address(mockEvaluatorSelector), newEvaluatorSelector);
        
        swarmCore.updateEvaluatorSelector(newEvaluatorSelector);
    }

    // ============================================================================
    // INTERFACE VALIDATION TESTS
    // ============================================================================

    function test_updateTrainerSelector_validatesInterfaceSupport() public {
        // Test that _updateTrainerSelector validates interface support
        address newTrainerSelector = address(invalidEvaluatorSelector);
        
        vm.expectRevert(SwarmCore.InvalidTrainerSelector.selector);
        swarmCore.updateTrainerSelector(newTrainerSelector);
    }

    function test_updateEvaluatorSelector_validatesInterfaceSupport() public {
        // Test that _updateEvaluatorSelector validates interface support
        address newEvaluatorSelector = address(invalidEvaluatorSelector);
        
        vm.expectRevert(SwarmCore.InvalidEvaluatorSelector.selector);
        swarmCore.updateEvaluatorSelector(newEvaluatorSelector);
    }

    function test_updateTrainerSelector_rejectsZeroAddress() public {
        // Test that _updateTrainerSelector rejects zero address
        vm.expectRevert(SwarmCore.InvalidTrainerSelector.selector);
        swarmCore.updateTrainerSelector(address(0));
    }

    function test_updateEvaluatorSelector_rejectsZeroAddress() public {
        // Test that _updateEvaluatorSelector rejects zero address
        vm.expectRevert(SwarmCore.InvalidEvaluatorSelector.selector);
        swarmCore.updateEvaluatorSelector(address(0));
    }

    function test_updateTrainerSelector_acceptsValidInterface() public {
        // Test that _updateTrainerSelector accepts valid interface implementation
        address newTrainerSelector = address(new AlwaysSampled());

        swarmCore.updateTrainerSelector(newTrainerSelector);
        
        address updatedSelector = swarmCore.getTrainerSelector();
        assertEq(updatedSelector, newTrainerSelector, "Should accept valid interface implementation");
    }

    function test_updateEvaluatorSelector_acceptsValidInterface() public {
        // Test that _updateEvaluatorSelector accepts valid interface implementation
        address newEvaluatorSelector = address(new MockSelector());

        swarmCore.updateEvaluatorSelector(newEvaluatorSelector);
        
        address updatedSelector = swarmCore.getEvaluatorSelector();
        assertEq(updatedSelector, newEvaluatorSelector, "Should accept valid interface implementation");
    }

    // ============================================================================
    // EDGE CASES AND ERROR HANDLING
    // ============================================================================

    function test_updateTrainerSelector_handlesNonERC165Contract() public {
        // Test that _updateTrainerSelector handles contracts that don't support ERC165
        address nonERC165Contract = makeAddr("nonERC165Contract");
        
        // Mock a revert when calling supportsInterface
        vm.mockCallRevert(
            nonERC165Contract,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            "Not supported"
        );
        
        vm.expectRevert(SwarmCore.InvalidTrainerSelector.selector);
        swarmCore.updateTrainerSelector(nonERC165Contract);
    }

    function test_updateEvaluatorSelector_handlesNonERC165Contract() public {
        // Test that _updateEvaluatorSelector handles contracts that don't support ERC165
        address nonERC165Contract = makeAddr("nonERC165Contract");
        
        // Mock a revert when calling supportsInterface
        vm.mockCallRevert(
            nonERC165Contract,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            "Not supported"
        );

        vm.expectRevert(SwarmCore.InvalidEvaluatorSelector.selector);
        swarmCore.updateEvaluatorSelector(nonERC165Contract);
    }

    function test_updateTrainerSelector_handlesInterfaceReturningFalse() public {
        // Test that _updateTrainerSelector handles contracts that return false for interface support
        address unsupportedContract = makeAddr("unsupportedContract");
        
        // Mock the contract to return false for ISelector interface
        vm.mockCall(
            unsupportedContract,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(false)
        );
        
        vm.expectRevert(SwarmCore.InvalidTrainerSelector.selector);
        swarmCore.updateTrainerSelector(unsupportedContract);
    }

    function test_updateEvaluatorSelector_handlesInterfaceReturningFalse() public {
        // Test that _updateEvaluatorSelector handles contracts that return false for interface support
        address unsupportedContract = makeAddr("unsupportedContract");
        
        // Mock the contract to return false for ISelector interface
        vm.mockCall(
            unsupportedContract,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(false)
        );

        vm.expectRevert(SwarmCore.InvalidEvaluatorSelector.selector);
        swarmCore.updateEvaluatorSelector(unsupportedContract);
    }

    // ============================================================================
    // CONTRIBUTION CALCULATOR TESTS
    // ============================================================================

    function test_updateContributionCalculator_emitsEvent() public {
        // Test that _updateContributionCalculator emits the correct event
        address newContributionCalculator = address(new MockContributionCalculator());
        
        vm.expectEmit(true, true, false, false);
        emit ContributionCalculatorUpdated(address(mockContributionCalculator), newContributionCalculator);
        
        swarmCore.updateContributionCalculator(newContributionCalculator);
    }

    function test_updateContributionCalculator_validatesInterfaceSupport() public {
        // Test that _updateContributionCalculator validates interface support
        address newContributionCalculator = address(invalidEvaluatorSelector);
        
        vm.expectRevert(SwarmCore.InvalidContributionCalculator.selector);
        swarmCore.updateContributionCalculator(newContributionCalculator);
    }

    function test_updateContributionCalculator_rejectsZeroAddress() public {
        // Test that _updateContributionCalculator rejects zero address
        vm.expectRevert(SwarmCore.InvalidContributionCalculator.selector);
        swarmCore.updateContributionCalculator(address(0));
    }

    function test_updateContributionCalculator_acceptsValidInterface() public {
        // Test that _updateContributionCalculator accepts valid interface implementation
        address newContributionCalculator = address(new MockContributionCalculator());

        swarmCore.updateContributionCalculator(newContributionCalculator);
        
        address updatedCalculator = swarmCore.getContributionCalculator();
        assertEq(updatedCalculator, newContributionCalculator, "Should accept valid interface implementation");
    }

    function test_updateTrainerSelector_preservesPreviousSelector() public {
        // Test that _updateTrainerSelector correctly tracks the previous selector
        address firstSelector = makeAddr("firstSelector");
        address secondSelector = makeAddr("secondSelector");
        
        // Mock both selectors to support ISelector interface
        vm.mockCall(
            firstSelector,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(true)
        );
        vm.mockCall(
            secondSelector,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(true)
        );

        // First update
        vm.expectEmit(true, true, false, false);
        emit TrainerSelectorUpdated(address(alwaysSampled), firstSelector);
        swarmCore.updateTrainerSelector(firstSelector);
        
        // Second update - should emit previous selector as firstSelector
        vm.expectEmit(true, true, false, false);
        emit TrainerSelectorUpdated(firstSelector, secondSelector);
        swarmCore.updateTrainerSelector(secondSelector);
    }

    function test_updateEvaluatorSelector_preservesPreviousSelector() public {
        // Test that _updateEvaluatorSelector correctly tracks the previous selector
        address firstSelector = makeAddr("firstSelector");
        address secondSelector = makeAddr("secondSelector");
        
        // Mock both selectors to support ISelector interface
        vm.mockCall(
            firstSelector,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(true)
        );
        vm.mockCall(
            secondSelector,
            abi.encodeWithSelector(IERC165.supportsInterface.selector, type(ISelector).interfaceId),
            abi.encode(true)
        );

        // First update
        vm.expectEmit(true, true, false, false);
        emit EvaluatorSelectorUpdated(address(mockEvaluatorSelector), firstSelector);
        swarmCore.updateEvaluatorSelector(firstSelector);
        
        // Second update - should emit previous selector as firstSelector
        vm.expectEmit(true, true, false, false);
        emit EvaluatorSelectorUpdated(firstSelector, secondSelector);
        swarmCore.updateEvaluatorSelector(secondSelector);
    }

    // ============================================================================
    // STORAGE AND STATE TESTS
    // ============================================================================

    function test_storageIsolation() public {
        // Test that storage is properly isolated using namespaced storage
        address newTrainerSelector = address(new AlwaysSampled());
        address newEvaluatorSelector = address(new MockSelector());

        // Update both selectors
        swarmCore.updateTrainerSelector(newTrainerSelector);
        swarmCore.updateEvaluatorSelector(newEvaluatorSelector);
        
        // Verify the storage was updated
        address storedTrainerSelector = swarmCore.getTrainerSelector();
        address storedEvaluatorSelector = swarmCore.getEvaluatorSelector();
        assertEq(storedTrainerSelector, newTrainerSelector, "Trainer selector storage should be properly updated");
        assertEq(storedEvaluatorSelector, newEvaluatorSelector, "Evaluator selector storage should be properly updated");
        
        // Verify the original implementation storage is unchanged
        address implementationTrainerSelector = implementation.getTrainerSelector();
        address implementationEvaluatorSelector = implementation.getEvaluatorSelector();
        assertEq(implementationTrainerSelector, address(0), "Implementation trainer selector storage should remain unchanged");
        assertEq(implementationEvaluatorSelector, address(0), "Implementation evaluator selector storage should remain unchanged");
    }

    function test_initializationState() public view {
        // Test that the contract is properly initialized after setup
        // This test verifies that the initializer modifier worked correctly
        address trainerSelector = swarmCore.getTrainerSelector();
        address evaluatorSelector = swarmCore.getEvaluatorSelector();
        assertEq(trainerSelector, address(alwaysSampled), "Contract should be properly initialized with trainer selector");
        assertEq(evaluatorSelector, address(mockEvaluatorSelector), "Contract should be properly initialized with evaluator selector");
    }
}
