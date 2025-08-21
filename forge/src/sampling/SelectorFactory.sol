// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Ownable} from "@openzeppelin-contracts-5.2.0/access/Ownable.sol";
import {ISelector} from "./ISelector.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

/// @title SelectorFactory
/// @notice Factory contract for creating selector instances
/// @dev Allows admins to register selector implementations and users to create instances
contract SelectorFactory is Ownable {
    /// @notice Emitted when a new selector implementation is registered
    /// @param id The unique identifier for the selector implementation
    /// @param implementation The address of the selector implementation contract
    event SelectorImplementationRegistered(bytes32 indexed id, address indexed implementation);

    /// @notice Emitted when a new selector instance is created
    /// @param id The identifier of the selector implementation used
    /// @param instance The address of the newly created selector instance
    /// @param creator The address that created the instance
    event SelectorInstanceCreated(bytes32 indexed id, address indexed instance, address indexed creator);

    /// @notice Emitted when a selector implementation is updated
    /// @param id The unique identifier for the selector implementation
    /// @param oldImplementation The previous implementation address
    /// @param newImplementation The new implementation address
    event SelectorImplementationUpdated(bytes32 indexed id, address indexed oldImplementation, address indexed newImplementation);

    /// @notice Emitted when a selector implementation is removed
    /// @param id The unique identifier for the selector implementation
    /// @param implementation The address of the removed implementation
    event SelectorImplementationRemoved(bytes32 indexed id, address indexed implementation);

    /// @notice Custom errors for better gas efficiency
    error SelectorImplementationNotFound();
    error SelectorImplementationAlreadyExists();
    error SelectorImplementationInvalid();
    error SelectorCreationFailed();

    /// @notice Mapping from selector ID to implementation address
    mapping(bytes32 => address) public selectorImplementations;

    /// @notice Constructor sets the initial owner
    /// @param initialOwner The initial owner of the contract
    constructor(address initialOwner) Ownable(initialOwner) {}

    /// @notice Register a new selector implementation
    /// @dev Only callable by the owner
    /// @param implementation The address of the selector implementation contract
    function registerSelectorImplementation(address implementation) external onlyOwner {
        // Verify the implementation supports ISelector interface
        if (!_supportsISelector(implementation)) {
            revert SelectorImplementationInvalid();
        }
        ISelector selector = ISelector(implementation);
        // aderyn-fp-next-line(reentrancy-state-change)
        (,, string memory version,,,,) = selector.eip712Domain();
        bytes32 id = getID(version);

        if (isSelectorRegistered(id)) {
            revert SelectorImplementationAlreadyExists();
        }


        selectorImplementations[id] = implementation;

        emit SelectorImplementationRegistered(id, implementation);
    }

    function isSelectorRegistered(bytes32 id) public view returns (bool) {
        return selectorImplementations[id] != address(0);
    }

    function isSelectorVersionRegistered(string memory version) external view returns (bool) {
        bytes32 id = getID(version);
        return isSelectorRegistered(id);
    }

    function getID(string memory version) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(version));
    }


    /// @notice Remove a selector implementation
    /// @dev Only callable by the owner
    /// @param id The unique identifier for the selector implementation
    function removeSelectorImplementation(bytes32 id) external onlyOwner {
        if (!isSelectorRegistered(id)) {
            revert SelectorImplementationNotFound();
        }

        address implementation = selectorImplementations[id];
        
        // Remove from mappings
        delete selectorImplementations[id];

        emit SelectorImplementationRemoved(id, implementation);
    }

    /// @notice Create a new selector instance using UUPS proxy
    /// @param id The identifier of the selector implementation to use
    /// @param salt The salt for CREATE2 deployment
    /// @param initData The encoded initialization data for the selector instance
    /// @return instance The address of the newly created selector instance
    function createSelector(bytes32 id, bytes32 salt, bytes memory initData) external returns (address instance) {
        if (!isSelectorRegistered(id)) {
            revert SelectorImplementationNotFound();
        }

        address implementation = selectorImplementations[id];

        // Create new UUPS proxy instance using ERC1967Proxy
        // The proxy will delegate all calls to the implementation
        ERC1967Proxy proxy = new ERC1967Proxy{salt: salt}(
            implementation,
            initData
        );
        
        instance = address(proxy);

        if (instance == address(0)) {
            revert SelectorCreationFailed();
        }

        emit SelectorInstanceCreated(id, instance, msg.sender);

        return instance;
    }

    /// @notice Get the implementation address for a selector ID
    /// @param id The selector ID
    /// @return The implementation address
    function getSelectorImplementation(bytes32 id) external view returns (address) {
        return selectorImplementations[id];
    }

    /// @notice Check if an address supports the ISelector interface
    /// @param addr The address to check
    /// @return True if the address supports ISelector interface, false otherwise
    function _supportsISelector(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(ISelector).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }
}
