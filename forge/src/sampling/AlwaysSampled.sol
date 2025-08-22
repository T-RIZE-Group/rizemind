// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ISelector} from "./ISelector.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";
import {EIP712Upgradeable} from "@openzeppelin-contracts-upgradeable-5.2.0/utils/cryptography/EIP712Upgradeable.sol";

contract AlwaysSampled is ISelector, IERC165, EIP712Upgradeable {

    string private constant _VERSION = "always-sampled-v1.0.0";

    function initialize() external initializer {
        __EIP712_init("AlwaysSampled", _VERSION);
    }

    function isSelected(
        address,
        uint256
    ) external pure override virtual returns (bool) {
      return true;
    }

    /**
     * @dev The version parameter for the EIP712 domain.
     */
    // solhint-disable-next-line func-name-mixedcase
    function _EIP712Version()
        internal
        pure
        virtual
        override(EIP712Upgradeable)
        returns (string memory)
    {
        return _VERSION;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) external pure override virtual returns (bool) {
        return interfaceId == type(ISelector).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }
}