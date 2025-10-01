// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Ownable} from "@openzeppelin-contracts-5.2.0/access/Ownable.sol";
import {ICompensation} from "./types.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

/// @title CompensationFactory
/// @notice Factory contract for creating compensation instances
/// @dev Allows admins to register compensation implementations and users to create instances
contract CompensationFactory is Ownable {
    /// @notice Emitted when a new compensation implementation is registered
    /// @param id The unique identifier for the compensation implementation
    /// @param implementation The address of the compensation implementation contract
    event CompensationImplementationRegistered(bytes32 indexed id, address indexed implementation);

    /// @notice Emitted when a new compensation instance is created
    /// @param id The identifier of the compensation implementation used
    /// @param instance The address of the newly created compensation instance
    /// @param creator The address that created the instance
    event CompensationInstanceCreated(bytes32 indexed id, address indexed instance, address indexed creator);

    /// @notice Emitted when a compensation implementation is updated
    /// @param id The unique identifier for the compensation implementation
    /// @param oldImplementation The previous implementation address
    /// @param newImplementation The new implementation address
    event CompensationImplementationUpdated(bytes32 indexed id, address indexed oldImplementation, address indexed newImplementation);

    /// @notice Emitted when a compensation implementation is removed
    /// @param id The unique identifier for the compensation implementation
    /// @param implementation The address of the removed implementation
    event CompensationImplementationRemoved(bytes32 indexed id, address indexed implementation);

    /// @notice Custom errors for better gas efficiency
    error CompensationImplementationNotFound();
    error CompensationImplementationAlreadyExists();
    error CompensationImplementationInvalid();
    error CompensationCreationFailed();

    /// @notice Mapping from compensation ID to implementation address
    mapping(bytes32 => address) public compensationImplementations;

    /// @notice Constructor sets the initial owner
    /// @param initialOwner The initial owner of the contract
    constructor(address initialOwner) Ownable(initialOwner) {}

    /// @notice Register a new compensation implementation
    /// @dev Only callable by the owner
    /// @param implementation The address of the compensation implementation contract
    function registerCompensationImplementation(address implementation) external onlyOwner {
        // Verify the implementation supports ICompensation interface
        if (!_supportsICompensation(implementation)) {
            revert CompensationImplementationInvalid();
        }
        ICompensation compensation = ICompensation(implementation);
        // aderyn-fp-next-line(reentrancy-state-change)
        (,, string memory version,,,,) = compensation.eip712Domain();
        bytes32 id = getID(version);

        if (isCompensationRegistered(id)) {
            revert CompensationImplementationAlreadyExists();
        }

        compensationImplementations[id] = implementation;

        emit CompensationImplementationRegistered(id, implementation);
    }

    function isCompensationRegistered(bytes32 id) public view returns (bool) {
        return compensationImplementations[id] != address(0);
    }

    function isCompensationVersionRegistered(string memory version) external view returns (bool) {
        bytes32 id = getID(version);
        return isCompensationRegistered(id);
    }

    function getID(string memory version) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(version));
    }

    /// @notice Remove a compensation implementation
    /// @dev Only callable by the owner
    /// @param id The unique identifier for the compensation implementation
    function removeCompensationImplementation(bytes32 id) external onlyOwner {
        if (!isCompensationRegistered(id)) {
            revert CompensationImplementationNotFound();
        }

        address implementation = compensationImplementations[id];
        
        // Remove from mappings
        delete compensationImplementations[id];

        emit CompensationImplementationRemoved(id, implementation);
    }

    /// @notice Create a new compensation instance using UUPS proxy
    /// @param id The identifier of the compensation implementation to use
    /// @param salt The salt for CREATE2 deployment
    /// @param initData The encoded initialization data for the compensation instance
    /// @return instance The address of the newly created compensation instance
    function createCompensation(bytes32 id, bytes32 salt, bytes memory initData) external returns (address instance) {
        if (!isCompensationRegistered(id)) {
            revert CompensationImplementationNotFound();
        }

        address implementation = compensationImplementations[id];

        // Create new UUPS proxy instance using ERC1967Proxy
        // The proxy will delegate all calls to the implementation
        ERC1967Proxy proxy = new ERC1967Proxy{salt: salt}(
            implementation,
            initData
        );
        
        instance = address(proxy);

        if (instance == address(0)) {
            revert CompensationCreationFailed();
        }

        emit CompensationInstanceCreated(id, instance, msg.sender);

        return instance;
    }

    /// @notice Get the implementation address for a compensation ID
    /// @param id The compensation ID
    /// @return The implementation address
    function getCompensationImplementation(bytes32 id) external view returns (address) {
        return compensationImplementations[id];
    }

    /// @notice Check if an address supports the ICompensation interface
    /// @param addr The address to check
    /// @return True if the address supports ICompensation interface, false otherwise
    function _supportsICompensation(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(ICompensation).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }
}
