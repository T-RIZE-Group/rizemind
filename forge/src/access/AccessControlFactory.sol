// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Ownable} from "@openzeppelin-contracts-5.2.0/access/Ownable.sol";
import {IAccessControl} from "./IAccessControl.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

/// @title AccessControlFactory
/// @notice Factory contract for creating access control instances
/// @dev Allows admins to register access control implementations and users to create instances
contract AccessControlFactory is Ownable {
    /// @notice Emitted when a new access control implementation is registered
    /// @param id The unique identifier for the access control implementation
    /// @param implementation The address of the access control implementation contract
    event AccessControlImplementationRegistered(bytes32 indexed id, address indexed implementation);

    /// @notice Emitted when a new access control instance is created
    /// @param id The identifier of the access control implementation used
    /// @param instance The address of the newly created access control instance
    /// @param creator The address that created the instance
    event AccessControlInstanceCreated(bytes32 indexed id, address indexed instance, address indexed creator);

    /// @notice Emitted when an access control implementation is updated
    /// @param id The unique identifier for the access control implementation
    /// @param oldImplementation The previous implementation address
    /// @param newImplementation The new implementation address
    event AccessControlImplementationUpdated(bytes32 indexed id, address indexed oldImplementation, address indexed newImplementation);

    /// @notice Emitted when an access control implementation is removed
    /// @param id The unique identifier for the access control implementation
    /// @param implementation The address of the removed implementation
    event AccessControlImplementationRemoved(bytes32 indexed id, address indexed implementation);

    /// @notice Custom errors for better gas efficiency
    error AccessControlImplementationNotFound();
    error AccessControlImplementationAlreadyExists();
    error AccessControlImplementationInvalid();
    error AccessControlCreationFailed();

    /// @notice Mapping from access control ID to implementation address
    mapping(bytes32 => address) public accessControlImplementations;

    /// @notice Constructor sets the initial owner
    /// @param initialOwner The initial owner of the contract
    constructor(address initialOwner) Ownable(initialOwner) {}

    /// @notice Register a new access control implementation
    /// @dev Only callable by the owner
    /// @param implementation The address of the access control implementation contract
    function registerAccessControlImplementation(address implementation) external onlyOwner {
        // Verify the implementation supports IAccessControl interface
        if (!_supportsIAccessControl(implementation)) {
            revert AccessControlImplementationInvalid();
        }
        IAccessControl accessControl = IAccessControl(implementation);
        // aderyn-fp-next-line(reentrancy-state-change)
        (,, string memory version,,,,) = accessControl.eip712Domain();
        bytes32 id = getID(version);

        if (isAccessControlRegistered(id)) {
            revert AccessControlImplementationAlreadyExists();
        }

        accessControlImplementations[id] = implementation;

        emit AccessControlImplementationRegistered(id, implementation);
    }

    function isAccessControlRegistered(bytes32 id) public view returns (bool) {
        return accessControlImplementations[id] != address(0);
    }

    function isAccessControlVersionRegistered(string memory version) external view returns (bool) {
        bytes32 id = getID(version);
        return isAccessControlRegistered(id);
    }

    function getID(string memory version) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(version));
    }

    /// @notice Remove an access control implementation
    /// @dev Only callable by the owner
    /// @param id The unique identifier for the access control implementation
    function removeAccessControlImplementation(bytes32 id) external onlyOwner {
        if (!isAccessControlRegistered(id)) {
            revert AccessControlImplementationNotFound();
        }

        address implementation = accessControlImplementations[id];
        
        // Remove from mappings
        delete accessControlImplementations[id];

        emit AccessControlImplementationRemoved(id, implementation);
    }

    /// @notice Create a new access control instance using UUPS proxy
    /// @param id The identifier of the access control implementation to use
    /// @param salt The salt for CREATE2 deployment
    /// @param initData The encoded initialization data for the access control instance
    /// @return instance The address of the newly created access control instance
    function createAccessControl(bytes32 id, bytes32 salt, bytes memory initData) external returns (address instance) {
        if (!isAccessControlRegistered(id)) {
            revert AccessControlImplementationNotFound();
        }

        address implementation = accessControlImplementations[id];

        // Create new UUPS proxy instance using ERC1967Proxy
        // The proxy will delegate all calls to the implementation
        ERC1967Proxy proxy = new ERC1967Proxy{salt: salt}(
            implementation,
            initData
        );
        
        instance = address(proxy);

        if (instance == address(0)) {
            revert AccessControlCreationFailed();
        }

        emit AccessControlInstanceCreated(id, instance, msg.sender);

        return instance;
    }

    /// @notice Get the implementation address for an access control ID
    /// @param id The access control ID
    /// @return The implementation address
    function getAccessControlImplementation(bytes32 id) external view returns (address) {
        return accessControlImplementations[id];
    }

    /// @notice Check if an address supports the IAccessControl interface
    /// @param addr The address to check
    /// @return True if the address supports IAccessControl interface, false otherwise
    function _supportsIAccessControl(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(IAccessControl).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }
}
