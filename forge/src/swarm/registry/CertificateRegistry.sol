// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {ICertificateRegistry} from "./ICertificateRegistry.sol";

abstract contract CertificateRegistry is ICertificateRegistry {
    mapping(bytes32 => bytes) private _certificates;

    event NewCertificate(bytes32 indexed id, bytes certificate);

    // aderyn-ignore-next-line(empty-block)
    function __CertificateRegistry_init() public {}

    function _setCertificate(bytes32 id, bytes calldata certificate) internal {
        _certificates[id] = certificate;
        emit NewCertificate(id, certificate);
    }

    function getCertificate(bytes32 id) public view returns (bytes memory) {
        return _certificates[id];
    }

    function supportsInterface(
        bytes4 interfaceId
    ) public view virtual returns (bool) {
        return
            interfaceId == ICertificateRegistry.getCertificate.selector ||
            interfaceId == ICertificateRegistry.setCertificate.selector;
    }
}
