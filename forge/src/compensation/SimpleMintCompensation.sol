// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import {CompensationSent} from "./types.sol";
import {ERC20Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/token/ERC20/ERC20Upgradeable.sol";
import {ICompensation} from "./types.sol";
import {EIP712} from "@openzeppelin-contracts-5.2.0/utils/cryptography/EIP712.sol";
import {AccessControlUpgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/access/AccessControlUpgradeable.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";

contract SimpleMintCompensation is ERC20Upgradeable, ICompensation, EIP712, AccessControlUpgradeable {

    string private constant _VERSION = "simple-mint-compensation-v1.0.0";
    uint8 constant CONTRIBUTION_DECIMALS = 6;
    
    /// @dev Storage namespace for SimpleMintCompensation
    struct SimpleMintCompensationStorage {
        uint256 targetRewardsPerRound;
        mapping(uint256 => uint256) totalContributions;
    }
    
    // Storage slot for SimpleMintCompensation namespace
    bytes32 private constant SIMPLE_MINT_COMPENSATION_STORAGE_SLOT = keccak256("SimpleMintCompensation.storage");

    bytes32 public constant MINTER_ROLE = keccak256("MINTER_ROLE");
    error BadRewards();

    constructor() EIP712("SimpleMintCompensation", _VERSION) {}

    function initialize(
        string memory name,
        string memory symbol,
        uint256 targetRewards,
        address initialAdmin,
        address minter
    ) external initializer {
        __SimpleMintCompensation_init(name, symbol, targetRewards);
        __AccessControl_init();
        _grantRole(DEFAULT_ADMIN_ROLE, initialAdmin);
        _grantRole(MINTER_ROLE, minter);
    }

    function __SimpleMintCompensation_init(
        string memory name,
        string memory symbol,
        uint256 maxRewards
    ) internal onlyInitializing {
        __ERC20_init(name, symbol);
        SimpleMintCompensationStorage storage $ = _getSimpleMintCompensationStorage();
        $.targetRewardsPerRound = maxRewards;
    }

    function distribute(
        uint256 roundId,
        address[] memory recipients,
        uint64[] memory contributions
    ) external onlyRole(MINTER_ROLE) {
        _distribute(roundId, recipients, contributions);
    }

    function getTargetRewardsPerRound() external view returns (uint256) {
        SimpleMintCompensationStorage storage $ = _getSimpleMintCompensationStorage();
        return $.targetRewardsPerRound;
    }

    function _distribute(
        uint256 roundId,
        address[] memory recipients,
        uint64[] memory contributions
    ) internal {
        uint256 nTrainers = recipients.length;
        uint256 totalContributions = 0;
        if (nTrainers != contributions.length) {
            revert BadRewards();
        }
        for (uint16 i = 0; i < nTrainers; i++) {
            address recipient = recipients[i];
            uint64 contribution = contributions[i];
            totalContributions += contribution;
            uint256 rewards = _calculateRewards(contribution);
            _mint(recipient, rewards);
            emit CompensationSent(recipient, rewards);
        }
        SimpleMintCompensationStorage storage $ = _getSimpleMintCompensationStorage();
        $.totalContributions[roundId] += totalContributions;
    }

    /**
     * @notice Returns a pointer to the storage namespace
     * @dev This function provides access to the namespaced storage
     */
    function _getSimpleMintCompensationStorage() private pure returns (SimpleMintCompensationStorage storage $) {
        bytes32 slot = SIMPLE_MINT_COMPENSATION_STORAGE_SLOT;
        assembly {
            $.slot := slot
        }
    }

    function _calculateRewards(
        uint64 contribution
    ) internal view returns (uint256) {
        SimpleMintCompensationStorage storage $ = _getSimpleMintCompensationStorage();
        return
            (uint256(contribution) * $.targetRewardsPerRound) /
            (10 ** CONTRIBUTION_DECIMALS);
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) public virtual view override returns (bool) {
        return interfaceId == type(ICompensation).interfaceId || 
               interfaceId == type(IERC165).interfaceId ||
               super.supportsInterface(interfaceId);
    }
}
