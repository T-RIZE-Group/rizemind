// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface ICertificateRegistry {
    function getCertificate(bytes32 id) external view returns (bytes memory);

    function setCertificate(bytes32 id, bytes calldata value) external;
}
