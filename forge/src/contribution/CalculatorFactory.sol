// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Ownable} from "@openzeppelin-contracts-5.2.0/access/Ownable.sol";
import {ContributionCalculator} from "./ContributionCalculator.sol";
import {IContributionCalculator} from "./types.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {ERC1967Proxy} from "@openzeppelin-contracts-5.2.0/proxy/ERC1967/ERC1967Proxy.sol";

/// @title CalculatorFactory
/// @notice Factory contract for creating ContributionCalculator instances
/// @dev Allows admins to register calculator implementations and users to create instances
contract CalculatorFactory is Ownable {
    /// @notice Emitted when a new calculator implementation is registered
    /// @param id The unique identifier for the calculator implementation
    /// @param implementation The address of the calculator implementation contract
    event CalculatorImplementationRegistered(bytes32 indexed id, address indexed implementation);

    /// @notice Emitted when a new calculator instance is created
    /// @param id The identifier of the calculator implementation used
    /// @param instance The address of the newly created calculator instance
    /// @param creator The address that created the instance
    event CalculatorInstanceCreated(bytes32 indexed id, address indexed instance, address indexed creator);

    /// @notice Emitted when a calculator implementation is updated
    /// @param id The unique identifier for the calculator implementation
    /// @param oldImplementation The previous implementation address
    /// @param newImplementation The new implementation address
    event CalculatorImplementationUpdated(bytes32 indexed id, address indexed oldImplementation, address indexed newImplementation);

    /// @notice Emitted when a calculator implementation is removed
    /// @param id The unique identifier for the calculator implementation
    /// @param implementation The address of the removed implementation
    event CalculatorImplementationRemoved(bytes32 indexed id, address indexed implementation);

    /// @notice Custom errors for better gas efficiency
    error CalculatorImplementationNotFound();
    error CalculatorImplementationAlreadyExists();
    error CalculatorImplementationInvalid();
    error CalculatorCreationFailed();

    /// @notice Mapping from calculator ID to implementation address
    mapping(bytes32 => address) public calculatorImplementations;

    /// @notice Constructor sets the initial owner
    /// @param initialOwner The initial owner of the contract
    constructor(address initialOwner) Ownable(initialOwner) {}

    /// @notice Register a new calculator implementation
    /// @dev Only callable by the owner
    /// @param implementation The address of the calculator implementation contract
    function registerCalculatorImplementation(address implementation) external onlyOwner {
        // Verify the implementation supports ContributionCalculator interface
        if (!_supportsContributionCalculator(implementation)) {
            revert CalculatorImplementationInvalid();
        }
        
        // Get version from the implementation (assuming it has a version function)
        string memory version = _getCalculatorVersion(implementation);
        bytes32 id = getID(version);

        if (isCalculatorRegistered(id)) {
            revert CalculatorImplementationAlreadyExists();
        }

        calculatorImplementations[id] = implementation;

        emit CalculatorImplementationRegistered(id, implementation);
    }

    function isCalculatorRegistered(bytes32 id) public view returns (bool) {
        return calculatorImplementations[id] != address(0);
    }

    function isCalculatorVersionRegistered(string memory version) external view returns (bool) {
        bytes32 id = getID(version);
        return isCalculatorRegistered(id);
    }

    function getID(string memory version) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(version));
    }

    /// @notice Remove a calculator implementation
    /// @dev Only callable by the owner
    /// @param id The unique identifier for the calculator implementation
    function removeCalculatorImplementation(bytes32 id) external onlyOwner {
        if (!isCalculatorRegistered(id)) {
            revert CalculatorImplementationNotFound();
        }

        address implementation = calculatorImplementations[id];
        
        // Remove from mappings
        delete calculatorImplementations[id];

        emit CalculatorImplementationRemoved(id, implementation);
    }

    /// @notice Create a new calculator instance using UUPS proxy
    /// @param id The identifier of the calculator implementation to use
    /// @param salt The salt for CREATE2 deployment
    /// @param initData The initialization data for the calculator
    /// @return instance The address of the newly created calculator instance
    function createCalculator(bytes32 id, bytes32 salt, bytes memory initData) external returns (address instance) {
        if (!isCalculatorRegistered(id)) {
            revert CalculatorImplementationNotFound();
        }

        address implementation = calculatorImplementations[id];

        // Create new UUPS proxy instance using ERC1967Proxy
        // The proxy will delegate all calls to the implementation
        ERC1967Proxy proxy = new ERC1967Proxy{salt: salt}(
            implementation,
            initData
        );
        
        instance = address(proxy);

        if (instance == address(0)) {
            revert CalculatorCreationFailed();
        }

        emit CalculatorInstanceCreated(id, instance, msg.sender);

        return instance;
    }

    /// @notice Get the implementation address for a calculator ID
    /// @param id The calculator ID
    /// @return The implementation address
    function getCalculatorImplementation(bytes32 id) external view returns (address) {
        return calculatorImplementations[id];
    }

    /// @notice Check if an address supports the IContributionCalculator interface
    /// @param addr The address to check
    /// @return True if the address supports IContributionCalculator interface, false otherwise
    function _supportsContributionCalculator(address addr) internal view returns (bool) {
        if (addr == address(0)) {
            return false;
        }

        try IERC165(addr).supportsInterface(type(IContributionCalculator).interfaceId) returns (bool supported) {
            return supported;
        } catch {
            return false;
        }
    }

    /// @notice Get the version string from a calculator implementation
    /// @param implementation The implementation address
    /// @return version The version string
    function _getCalculatorVersion(address implementation) internal view returns (string memory version) {
        IContributionCalculator calculator = IContributionCalculator(implementation);
        // Get version from the EIP712 domain, similar to SelectorFactory
        (,, string memory domainVersion,,,,) = calculator.eip712Domain();
        return domainVersion;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(IERC165).interfaceId;
    }
}
