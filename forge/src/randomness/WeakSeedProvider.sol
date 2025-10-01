// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ISeedProvider} from "./ISeedProvider.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";
import {Initializable} from "@openzeppelin-contracts-upgradeable-5.2.0/proxy/utils/Initializable.sol";

/// @title WeakSeedProvider
/// @notice A simple seed provider that generates deterministic seeds based on contract address and round ID
/// @dev This is a "weak" seed provider as it uses predictable inputs for seed generation
contract WeakSeedProvider is ISeedProvider, IERC165, EIP712Upgradeable {
    /// @dev The version parameter for the EIP712 domain
    string private constant _VERSION = "weak-seed-provider-v1.0.0";

    /// @notice Initializes the contract
    /// @dev This function can only be called once during proxy deployment
    function initialize() external virtual initializer {
        __WeakSeedProvider_init();
    }

    function __WeakSeedProvider_init() internal onlyInitializing {
        __EIP712_init("WeakSeedProvider", _VERSION);
    }

    /// @notice Returns a seed for the RNG system
    /// @dev Generates a deterministic seed by hashing the contract address concatenated with the round ID
    /// @param roundId The round ID to get a seed for
    /// @return A 32-byte seed for the RNG system
    function getSeed(uint256 roundId) external view override returns (bytes32) {
        return keccak256(abi.encodePacked(address(this), roundId));
    }

    /// @dev The version parameter for the EIP712 domain
    function _EIP712Version()
        internal
        pure
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) external pure returns (bool) {
        return interfaceId == type(ISeedProvider).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }
}
