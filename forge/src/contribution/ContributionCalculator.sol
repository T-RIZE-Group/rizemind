// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";
import {UUPSUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/UUPSUpgradeable.sol";
import {AccessControlUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/access/AccessControlUpgradeable.sol";
import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";
import {ShapleyValueCalculator} from "./ShapleyValueCalculator.sol";
import {IContributionCalculator} from "./types.sol";

/**
 * @title ContributionCalculator
 * @dev UUPS upgradeable contract for calculating Shapley values with admin-controlled result registration
 * @notice This contract implements ShapleyValueCalculator and provides admin-only access to register results
 */
contract ContributionCalculator is 
    Initializable, 
    UUPSUpgradeable, 
    AccessControlUpgradeable, 
    EIP712Upgradeable,
    ShapleyValueCalculator,
    IContributionCalculator
{
    string private constant _VERSION = "contribution-calculator-v1.0.0";

    // Events
    event ContributionResultRegistered(
        uint256 indexed roundId,
        uint256 indexed setId,
        bytes32 modelHash,
        int256 result,
        address indexed admin
    );

    event AdminRoleGranted(address indexed account, address indexed admin);
    event AdminRoleRevoked(address indexed account, address indexed admin);

    // Errors
    error Unauthorized();
    error InvalidUpgrade();

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /**
     * @dev Initializes the contract with the initial admin
     * @param initialAdmin The initial admin address
     */
    function initialize(address initialAdmin) external initializer {
        if (initialAdmin == address(0)) {
            revert InvalidParameters();
        }

        __UUPSUpgradeable_init();
        __AccessControl_init();
        __EIP712_init("ContributionCalculator", _VERSION);

        _grantRole(DEFAULT_ADMIN_ROLE, initialAdmin);
    }

    /**
     * @dev Register an evaluation result
     * @param roundId The round ID when evaluation was performed
     * @param setId The set ID for the evaluation
     * @param modelHash Hash of the model being evaluated
     * @param result The evaluation result
     */
    function registerResult(
        uint256 roundId,
        uint256 setId,
        bytes32 modelHash,
        int256 result
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {

        _registerResult(roundId, setId, modelHash, result);
        
        emit ContributionResultRegistered(roundId, setId, modelHash, result, msg.sender);
    }

    /**
     * @dev Grant admin role to an account
     * @param account The account to grant admin role to
     */
    function grantAdminRole(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (account == address(0)) {
            revert InvalidParameters();
        }
        
        _grantRole(DEFAULT_ADMIN_ROLE, account);
        emit AdminRoleGranted(account, msg.sender);
    }

    /**
     * @dev Revoke admin role from an account
     * @param account The account to revoke admin role from
     */
    function revokeAdminRole(address account) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (account == address(0)) {
            revert InvalidParameters();
        }
        
        _revokeRole(DEFAULT_ADMIN_ROLE, account);
        emit AdminRoleRevoked(account, msg.sender);
    }

    /**
     * @dev Calculate Shapley value for a specific trainer in a round
     * @param roundId The round ID
     * @param trainerIndex The trainer index
     * @param numberOfTrainers The number of trainers in that round
     * @return The calculated Shapley value
     */
    function calculateContribution(uint256 roundId, uint256 trainerIndex, uint8 numberOfTrainers) external view returns (int256) {
        return _calcShapley(roundId, trainerIndex, numberOfTrainers);
    }

    /**
     * @dev Required by the OZ UUPS module
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(DEFAULT_ADMIN_ROLE) {
        if (newImplementation == address(0)) {
            revert InvalidUpgrade();
        }
    }

    /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) public view override(AccessControlUpgradeable, ShapleyValueCalculator) virtual returns (bool) {
        return super.supportsInterface(interfaceId) || 
               interfaceId == type(IContributionCalculator).interfaceId;
    }
}
