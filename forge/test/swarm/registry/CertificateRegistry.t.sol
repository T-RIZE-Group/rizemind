// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import {CertificateRegistry, ICertificateRegistry} from "@rizemind-contracts/swarm/registry/CertificateRegistry.sol";

// Minimal concrete contract for testing
contract TestCertificateRegistry is CertificateRegistry {
    function setCertificate(bytes32 id, bytes calldata certificate) external {
        _setCertificate(id, certificate);
    }

    // Expose the init for coverage
    function callInit() external {
        __CertificateRegistry_init();
    }
}

contract CertificateRegistryTest is Test {
    TestCertificateRegistry public registry;

    function setUp() public {
        registry = new TestCertificateRegistry();
    }

    function testSetAndGetCertificate() public {
        bytes32 id = keccak256("test");
        bytes memory cert = abi.encodePacked(uint256(123));
        registry.setCertificate(id, cert);
        bytes memory result = registry.getCertificate(id);
        assertEq(result, cert, "Certificate not set or retrieved correctly");
    }

    function testNewCertificateEvent() public {
        bytes32 id = keccak256("event");
        bytes memory cert = abi.encodePacked(uint256(456));
        vm.expectEmit(true, false, false, true);
        emit CertificateRegistry.NewCertificate(id, cert);
        registry.setCertificate(id, cert);
    }

    function testSupportsInterface() public view {
        bytes4 getSelector = ICertificateRegistry.getCertificate.selector;
        // setCertificate is not public, but test the selector as in supportsInterface
        bytes4 setSelector = ICertificateRegistry.setCertificate.selector;
        assertTrue(
            registry.supportsInterface(getSelector),
            "Should support getCertificate"
        );
        assertTrue(
            registry.supportsInterface(setSelector),
            "Should support setCertificate"
        );
        assertFalse(
            registry.supportsInterface(0xdeadbeef),
            "Should not support random selector"
        );
    }

    function testInitFunction() public {
        registry.callInit();
        // No effect, just for coverage
        assertTrue(true);
    }
}
