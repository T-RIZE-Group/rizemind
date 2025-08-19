// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ISelector} from "./ISelector.sol";
import {IERC165} from "@openzeppelin-contracts-5.2.0/utils/introspection/IERC165.sol";

contract AlwaysSampled is ISelector, IERC165 {
    function isSelected(
        address,
        uint256
    ) external pure override returns (bool) {
      return true;
    }

    /// @dev See {IERC165-supportsInterface}
    function supportsInterface(bytes4 interfaceId) external pure override returns (bool) {
        return interfaceId == type(ISelector).interfaceId || 
               interfaceId == type(IERC165).interfaceId;
    }
}