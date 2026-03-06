# Wallet-Key Client + AWS Nitro Enclave Tutorial

This tutorial shows how to run Rizemind TEE aggregation with:

- Normal clients (no client-side TEE)
- Wallet private keys for ECDH encryption
- A Nitro Enclave for decryption and aggregation

The goal is reproducible understanding first, then production hardening later.

## What You Will Build

By the end, you will run:

1. Nitro host checks
2. EIF build from `src/py/rizemind/tee/Dockerfile.enclave`
3. Enclave startup + attestation/public key fetch
4. Wallet-key ECDH encryption walkthrough
5. End-to-end encrypted aggregation round-trip

## Important Key Fact

Wallet **address** is identity.
Wallet **private key** performs ECDH and decryption key derivation.

An address by itself cannot decrypt anything.

## Architecture Roles

- Client:
  Holds wallet private key, encrypts model updates for enclave.
- Parent EC2:
  Starts enclave, relays payloads over vsock, cannot inspect enclave memory.
- Enclave:
  Holds enclave private key, decrypts payloads, runs FedAvg, returns aggregate.

## Message Flow Mapped to Code

```text
1) Parent starts enclave
   - rizemind/tee/nitro/nitro_enclave.py:NitroTEEEnclave.initialize

2) Enclave generates key + attestation, listens on vsock
   - rizemind/tee/nitro/enclave_server.py:main
   - rizemind/tee/nitro/enclave_server.py:_get_nsm_attestation

3) Client receives enclave public key
   - rizemind/tee/tee_strategy.py:configure_fit

4) Client uses wallet private key for ECDH and AES-GCM encryption
   - rizemind/tee/tee_client_mod.py:tee_encryption_mod
   - rizemind/tee/crypto.py

5) Parent forwards encrypted updates to enclave over vsock
   - rizemind/tee/nitro/nitro_enclave.py:aggregate
   - rizemind/tee/vsock.py

6) Enclave decrypts each update and aggregates
   - rizemind/tee/nitro/enclave_server.py:_handle_aggregation
```

## Defaults Used in This Tutorial

- Region: `us-east-1`
- Parent instance: Nitro-capable EC2 with enclaves enabled
- Wallet source: deterministic test mnemonic via `eth_account`
- Key derivation: `ec_key_from_account` from `rizemind/tee/crypto.py`

## Prerequisites

Run the tutorial from a Nitro-capable EC2 host with enclave support enabled.

- `nitro-cli` installed
- Docker installed/running
- Nitro allocator service running
- Current user is in `ne` and `docker` groups

## Step 0: Repo Setup

Purpose:
Prepare repo and dependencies.

Command:

```bash
cd /path/to/rizemind
uv sync --no-group
chmod +x scripts/tee/tutorial/*.sh
```

Expected output:

- Dependency install succeeds.
- Shell scripts become executable.

Verification checkpoint:

- `ls -l scripts/tee/tutorial/*.sh` shows executable bit.

## Step 1: Nitro Host Readiness Check

Purpose:
Fail fast if host cannot run Nitro Enclaves.

Command:

```bash
scripts/tee/tutorial/01_prereq_check.sh
```

Expected output:

- PASS lines for `nitro-cli`, Docker, allocator service, and user groups.
- `nitro-cli describe-enclaves` reports no running enclaves.

Verification checkpoint:

- Script exits `0`.

## Step 2: Build EIF from Current Enclave Code

Purpose:
Create enclave image from repository source.

Command:

```bash
scripts/tee/tutorial/02_build_eif.sh
```

Expected output:

- Docker image `rizemind-enclave` built.
- `enclave.eif` created at repo root (default).
- PCR values written to `results/nitro_pcrs.txt`.

Verification checkpoint:

- `test -f enclave.eif`
- `cat results/nitro_pcrs.txt` contains `PCR0`, `PCR1`, `PCR2`.

## Step 3: Start Enclave and Fetch Attestation/Public Key

Purpose:
Launch Nitro enclave and confirm key/attestation retrieval.

Command:

```bash
uv run python scripts/tee/tutorial/03_run_enclave_and_fetch_attestation.py
```

Expected output:

- Enclave ID and CID printed.
- Public key length and attestation length printed.
- Artifact written to `results/nitro_attestation.json`.

Verification checkpoint:

- Public key length > 0
- Attestation length > 0

Note:
By default this script terminates the enclave after fetching data.
Use `--keep-running` if you want it to remain active for manual debugging.

## Step 4: Wallet-Key ECDH Walkthrough (Client Side, No TEE)

Purpose:
Understand wallet-key cryptography in isolation.

Command:

```bash
uv run python scripts/tee/tutorial/04_wallet_ecdh_encrypt.py
```

Expected output:

- Wallet address and account path printed.
- Control round-trip decrypt succeeds.
- Ciphertext/nonce generated.
- Optional target encryption generated if `results/nitro_attestation.json` exists.

Verification checkpoint:

- Script prints control decrypt success and exits `0`.

## Step 5: End-to-End Aggregation Round Trip on Nitro

Purpose:
Prove encrypted multi-client updates are decrypted and aggregated inside enclave.

Command:

```bash
uv run python scripts/tee/tutorial/05_aggregate_round_trip.py
```

Expected output:

- Enclave starts.
- Two wallet-derived encrypted updates sent.
- Aggregated output equals expected vector `[4.0, 6.0]`.
- Result artifact written to `results/nitro_round_trip.json`.

Verification checkpoint:

- Script prints success and exits `0`.

## Step 6: Connect Tutorial to Real Strategy Wiring

This section shows wiring only. Do not copy blindly into production.

### Client Mods: Before

```python
from flwr.client import ClientApp
from rizemind.authentication import authentication_mod
from rizemind.authentication.notary.model import model_notary_mod
from rizemind.swarm.modules.contribution.register_mod import register_contribution_mod

app = ClientApp(
    client_fn,
    mods=[authentication_mod, register_contribution_mod, model_notary_mod],
)
```

### Client Mods: After (TEE Encryption Added)

```python
from flwr.client import ClientApp
from rizemind.authentication import authentication_mod
from rizemind.authentication.notary.model import model_notary_mod
from rizemind.swarm.modules.contribution.register_mod import register_contribution_mod
from rizemind.tee import tee_encryption_mod

app = ClientApp(
    client_fn,
    mods=[authentication_mod, register_contribution_mod, model_notary_mod, tee_encryption_mod],
)
```

### Server Strategy: Before

```python
base_strategy = FedAvg(...)
auth_strategy = EthAccountStrategy(base_strategy, swarm, account)
```

### Server Strategy: After (TEE Wrapper Added)

```python
from rizemind.tee import NitroAttestationVerifier, TEEAggregationStrategy
from rizemind.tee.nitro.nitro_enclave import NitroTEEEnclave

base_strategy = FedAvg(...)
tee_enclave = NitroTEEEnclave(eif_path="enclave.eif", cpu_count=2, memory_mib=2048)
tee_verifier = NitroAttestationVerifier()
tee_strategy = TEEAggregationStrategy(base_strategy, tee_enclave, tee_verifier)
auth_strategy = EthAccountStrategy(tee_strategy, swarm, account)
```

## Step 7: Real Remote Flow (Laptop -> EC2 -> Nitro)

This is the practical setup you asked for: model updates sent from your computer
to an EC2 parent that forwards them into Nitro enclave.

### 7.1 Start Relay on EC2

On EC2:

```bash
cd /path/to/rizemind
uv sync --no-group
scripts/tee/tutorial/02_build_eif.sh
uv run python scripts/tee/tutorial/06_nitro_relay_server.py \
  --host 127.0.0.1 \
  --port 8080 \
  --api-token "change-me-demo-token"
```

Expected:

- Relay starts.
- Enclave ID/CID printed.
- Public key and attestation lengths printed.

### 7.2 Open Secure Tunnel from Laptop

On your laptop (new terminal):

```bash
ssh -i /path/to/key.pem -L 8080:127.0.0.1:8080 ec2-user@<EC2_PUBLIC_IP>
```

Keep this terminal open.

### 7.3 Submit Two Real Encrypted Updates from Laptop

On your laptop (repo checkout):

```bash
cd /path/to/rizemind
uv sync --no-group

uv run python scripts/tee/tutorial/07_remote_wallet_submit.py \
  --relay-url http://127.0.0.1:8080 \
  --api-token "change-me-demo-token" \
  --round-id 1 \
  --account-index 0 \
  --weights "2.0,4.0" \
  --num-examples 100 \
  --metadata '{"source":"laptop-a"}'

uv run python scripts/tee/tutorial/07_remote_wallet_submit.py \
  --relay-url http://127.0.0.1:8080 \
  --api-token "change-me-demo-token" \
  --round-id 1 \
  --account-index 1 \
  --weights "6.0,8.0" \
  --num-examples 100 \
  --metadata '{"source":"laptop-a"}'
```

Expected:

- Both submits return PASS.
- Pending count increments on relay.

### 7.4 Trigger Enclave Aggregation and Read Result

On your laptop:

```bash
uv run python scripts/tee/tutorial/08_remote_aggregate.py \
  --relay-url http://127.0.0.1:8080 \
  --api-token "change-me-demo-token" \
  --round-id 1 \
  --min-updates 2
```

Expected:

- Aggregation succeeds.
- Decoded aggregated vector is `[[4.0, 6.0]]` for the above example.

Verification checkpoint:

- `results/remote_aggregate_result.json` exists with `used_updates >= 2`.

## Known Limitations (Current Codebase)

1. `NitroAttestationVerifier` currently performs structural checks but not full certificate-chain/signature cryptographic validation.
2. Enclave server falls back to placeholder attestation when NSM bindings are unavailable.
3. Evaluation-channel encryption (`configure_evaluate`) is not implemented yet.

## Test Cases Covered by This Tutorial

1. `TC1_PrereqHost`: Nitro host readiness script passes.
2. `TC2_EIFBuild`: EIF and PCR artifacts are generated.
3. `TC3_AttestationFetch`: Enclave returns public key + attestation.
4. `TC4_WalletECDH`: Wallet-derived key flow encrypts/decrypts in control round-trip.
5. `TC5_AggregationRoundTrip`: Encrypted updates aggregate to expected value.
6. `TC6_Cleanup`: No running enclave remains after scripts finish.
7. `TC7_RemoteRelayFlow`: Laptop submits encrypted updates over tunnel, enclave aggregates remotely.

## Troubleshooting Quick Tips

- If `nitro-cli` is missing:
  install Nitro Enclaves CLI packages and retry Step 1.
- If allocator check fails:
  start `nitro-enclaves-allocator.service` and confirm `/etc/nitro_enclaves/allocator.yaml`.
- If EIF build fails:
  check Docker daemon and free disk space, then rerun Step 2.
- If Step 5 fails on local machine:
  run on Nitro-capable EC2 only; vsock/Nitro features are not available on regular hosts.
