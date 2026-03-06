======================================
Minimal Nitro ECDH Number Demo
======================================

This guide builds a minimal end-to-end AWS Nitro Enclave flow:

1. Client encrypts ``{"number": N}`` with secp256k1 ECDH.
2. Enclave decrypts payload, computes ``N + 1``.
3. Enclave returns ``{"ok": true, "result": N+1}``.

The first step runs locally (no Nitro) to prove crypto correctness.

Prerequisites
=============

- Nitro-capable EC2 host with Nitro Enclaves enabled
- ``nitro-cli`` and Docker installed
- Allocator service running
- Python dependencies installed via ``uv sync``

Files
=====

- ``scripts/nitro_demo/01_init_keys.py``
- ``scripts/nitro_demo/02_local_roundtrip.py``
- ``scripts/nitro_demo/03_enclave_server.py``
- ``scripts/nitro_demo/04_parent_client.py``
- ``scripts/nitro_demo/05_build_eif.sh``
- ``scripts/nitro_demo/06_run_enclave.sh``
- ``scripts/nitro_demo/07_stop_enclave.sh``

Step 1: Generate/Re-use Keys
============================

.. code-block:: bash

   uv run python scripts/nitro_demo/01_init_keys.py

Run again to confirm key reuse (status should become ``reused``):

.. code-block:: bash

   uv run python scripts/nitro_demo/01_init_keys.py

Step 2: Local Crypto Round Trip (No Nitro)
==========================================

.. code-block:: bash

   uv run python scripts/nitro_demo/02_local_roundtrip.py --number 41

Expected response:

.. code-block:: json

   {"ok": true, "input": 41, "result": 42}

Optional negative test (tampered ciphertext):

.. code-block:: bash

   uv run python scripts/nitro_demo/02_local_roundtrip.py --number 41 --tamper

Step 3: Build EIF
=================

.. code-block:: bash

   chmod +x scripts/nitro_demo/05_build_eif.sh scripts/nitro_demo/06_run_enclave.sh scripts/nitro_demo/07_stop_enclave.sh
   scripts/nitro_demo/05_build_eif.sh

Step 4: Run Enclave
===================

.. code-block:: bash

   scripts/nitro_demo/06_run_enclave.sh

Capture ``EnclaveID`` and ``EnclaveCID`` from output.

Step 5: Send Encrypted Number to Enclave
========================================

.. code-block:: bash

   uv run python scripts/nitro_demo/04_parent_client.py --cid <ENCLAVE_CID> --number 99

Expected response:

.. code-block:: json

   {"ok": true, "result": 100}

Step 6: Stop Enclave
====================

.. code-block:: bash

   scripts/nitro_demo/07_stop_enclave.sh <ENCLAVE_ID>

Notes
=====

- This v1 intentionally skips attestation verification.
- Payload format is one integer field only.
- ``04_parent_client.py`` expects direct vsock connectivity from parent instance to enclave.
