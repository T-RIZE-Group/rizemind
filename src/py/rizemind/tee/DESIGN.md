# TEE-Based Secure Aggregation — Design Document

## Problem

In Rizemind's federated learning pipeline, three entities interact each round:

1. **Trainers** — train models locally and send updates to the aggregator
2. **Aggregator** — collects updates, runs FedAvg inside a TEE, produces an aggregated model
3. **Evaluators (Testers)** — receive the aggregated model, evaluate it, report scores back

Currently, model parameters flow in **plaintext** between all entities. While EIP-712 signatures verify _identity_ (who sent what), they don't provide _confidentiality_ — anyone handling the parameters can inspect the raw model weights. This is a problem because:

- Individual trainer updates may leak information about their private training data (model inversion, membership inference attacks).
- The aggregated model sent to evaluators is exposed to the aggregator operator, who could tamper with it before forwarding.
- A compromised aggregator node could selectively modify or exclude contributions.

## Solution

We use **Trusted Execution Environments (TEEs)** — specifically AWS Nitro Enclaves — combined with **Elliptic Curve Diffie-Hellman (ECDH) key exchange** so that every entity-to-entity channel is end-to-end encrypted, and aggregation happens inside a hardware-isolated enclave that no one (not even the server operator) can inspect.

### Why ECDH?

Diffie-Hellman key exchange lets two parties derive a shared secret over an untrusted channel without ever transmitting the secret itself. We use the **elliptic curve** variant (ECDH) on **secp256k1** because:

- The project already uses secp256k1 for Ethereum wallet keys and EIP-712 signatures.
- ECDH is fast (sub-millisecond per key exchange) — negligible overhead even with hundreds of trainers.
- **Persistent keys tied to blockchain identity**: trainers, evaluators, and the aggregator each generate their ECDH keypair once when they register on-chain. The same secp256k1 private key used for Ethereum signing (EIP-712) is reused for ECDH — no separate key management needed.

### Why AES-256-GCM?

After ECDH produces a shared secret, we derive a symmetric key via **HKDF-SHA256** and encrypt with **AES-256-GCM** because:

- Already used in the codebase (`rizemind.mnemonic.store`) for encrypting mnemonics.
- Authenticated encryption: GCM provides both confidentiality and integrity — tampered ciphertext is detected.
- Hardware-accelerated on all modern CPUs via AES-NI.

### Why AWS Nitro Enclaves?

- Isolated execution: the enclave has its own kernel, no shell, no SSH, no persistent storage. Communication only via vsock.
- Attestation: the Nitro Hypervisor cryptographically signs an attestation document proving what code is running inside the enclave (PCR0 = code hash). Clients can verify this before sending encrypted data.
- No operator access: even the EC2 instance owner cannot inspect enclave memory.

---

## System Control Flow Diagram

### Full Round Lifecycle

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ROUND LIFECYCLE                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 0: ENCLAVE INITIALIZATION (once, on first round)                      │
│                                                                             │
│   ┌──────────────────────┐         ┌──────────────────────────────────┐     │
│   │   Nitro Hypervisor   │         │       Nitro Enclave              │     │
│   │                      │ boot    │                                  │     │
│   │  nitro-cli           │────────►│  1. Generate ECDH keypair        │     │
│   │  run-enclave         │         │     (sk_enclave, pk_enclave)     │     │
│   │                      │         │                                  │     │
│   │                      │◄────────│  2. Request NSM attestation      │     │
│   │  Sign attestation    │         │     embedding pk_enclave         │     │
│   │  with Nitro root CA  │────────►│                                  │     │
│   │                      │         │  3. Listen on vsock:5000         │     │
│   └──────────────────────┘         └──────────────────────────────────┘     │
│                                           │                                 │
│                                     vsock │ GET_ATTESTATION                 │
│                                           ▼                                 │
│   ┌──────────────────────────────────────────────────────────────────┐      │
│   │  Aggregator (Parent EC2)                                         │      │
│   │  Receives: pk_enclave + attestation_document                     │      │
│   │  Verifies: COSE_Sign1 signature, cert chain, PCR0 measurement    │      │
│   └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: TRAINING — Trainers encrypt updates for TEE (Channel 1)            │
│                                                                [IMPLEMENTED]│
│                                                                             │
│   ┌────────────┐    configure_fit     ┌──────────────────────────────┐      │
│   │            │◄─────────────────────│  Aggregator                  │      │
│   │            │   FitIns.config:     │  TEEAggregationStrategy      │      │
│   │            │   • global params    │  .configure_fit()            │      │
│   │            │   • pk_enclave       │                              │      │
│   │  Trainer   │   • attestation_doc  └──────────────────────────────┘      │
│   │            │                                                            │
│   │            │   ── local training ──                                     │
│   │            │                                                            │
│   │            │   ── ECDH encryption ──                                    │
│   │            │   1. Generate ephemeral keypair (sk_i, pk_i)               │
│   │            │   2. shared_secret = ECDH(sk_i, pk_enclave)                │
│   │            │   3. sym_key = HKDF-SHA256(shared_secret)                  │
│   │            │   4. ciphertext, nonce = AES-GCM(sym_key, parameters)      │
│   │            │                                                            │
│   │            │    aggregate_fit     ┌──────────────────────────────┐      │
│   │            │─────────────────────►│  Aggregator                  │      │
│   │            │   FitRes.metrics:    │  TEEAggregationStrategy      │      │
│   └────────────┘   • ciphertext       │  .aggregate_fit()            │      │
│     (× N trainers) • nonce            │                              │      │
│                    • pk_i             │  Collects all encrypted      │      │
│                    • num_examples     │  updates, forwards to TEE    │      │
│                                       └──────────────┬───────────────┘      │
│                                                vsock │ AGGREGATE            │
│                                                      ▼                      │
│                                       ┌──────────────────────────────┐      │
│                                       │  Nitro Enclave               │      │
│                                       │                              │      │
│                                       │  For each trainer i:         │      │
│                                       │   shared = ECDH(sk_enc, pk_i)│      │
│                                       │   key = HKDF(shared)         │      │
│                                       │   params_i = AES-GCM-Dec(    │      │
│                                       │     key, nonce, ciphertext)  │      │
│                                       │                              │      │
│                                       │  FedAvg.aggregate_fit(       │      │
│                                       │    all decrypted params,     │      │
│                                       │    weighted by num_examples) │      │
│                                       │                              │      │
│                                       │  Return: aggregated_params   │      │
│                                       └──────────────┬───────────────┘      │
│                                                vsock │                      │
│                                                      ▼                      │
│                                       ┌──────────────────────────────┐      │
│                                       │  Aggregator receives         │      │
│                                       │  aggregated Parameters       │      │
│                                       └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: EVALUATION — Aggregator sends model to evaluators (Channel 2)      │
│                                                             [NOT YET IMPL'D]│
│                                                                             │
│   ┌──────────────────────────────┐   configure_evaluate    ┌─────────────┐  │
│   │  Aggregator                  │────────────────────────►│             │  │
│   │  TEEAggregationStrategy      │   EvalIns.config:       │             │  │
│   │  .configure_evaluate()       │   • pk_enclave          │  Evaluator  │  │
│   │                              │   • attestation_doc     │             │  │
│   │  For each evaluator j:       │   EvalIns.parameters:   │             │  │
│   │   pk_j = evaluator's pubkey  │   • encrypted model     │             │  │
│   │   shared = ECDH(sk_enc, pk_j)│     (ECDH with eval j)  │             │  │
│   │   key = HKDF(shared)         │                         │             │  │
│   │   encrypted = AES-GCM(       │                         │             │  │
│   │     key, aggregated_params)  │                         │             │  │
│   └──────────────────────────────┘                         │             │  │
│                                                            │  Evaluator  │  │
│                                                            │  decrypts:  │  │
│                                                            │  1. ECDH    │  │
│                                                            │     with    │  │
│                                                            │     pk_enc  │  │
│                                                            │  2. Decrypt │  │
│                                                            │     model   │  │
│                                                            │  3. Run     │  │
│                                                            │     eval    │  │
│                                                            │  4. Return  │  │
└─────────────────────────────────────────────────────────────────────────────┘


```

### Key Management Model

All three entities (trainers, evaluators, aggregator/TEE) are registered on the
blockchain. Each entity's ECDH key is derived from their Ethereum account's
secp256k1 private key — **generated once at registration, not per round**.

```
    ┌──────────────────────────────────────────────────────────┐
    │  ON-CHAIN REGISTRATION (one-time)                        │
    │                                                          │
    │  BIP-39 mnemonic  ──► HD derivation (m/44'/60'/0'/0/i)  │
    │       │                     │                            │
    │       ▼                     ▼                            │
    │  Ethereum Account     secp256k1 private key              │
    │  (address, signing)   (same key used for ECDH)           │
    │       │                     │                            │
    │       ▼                     ▼                            │
    │  EIP-712 signatures   ec_key_from_account(acct)          │
    │  (authentication)     (encryption via ECDH)              │
    └──────────────────────────────────────────────────────────┘

    Trainer:    sk_trainer, pk_trainer   ── registered on-chain
    Evaluator:  sk_eval,    pk_eval     ── registered on-chain
    TEE:        sk_enclave, pk_enclave  ── attested via Nitro, persistent per session
```

### ECDH Key Exchange Detail (Per Channel)

```
    Entity A (Trainer/Evaluator)              Entity B (TEE Enclave)
    ────────────────────────────              ──────────────────────

    Persistent keypair from                   Persistent keypair from
    Ethereum account:                         enclave initialization:
    sk_a (from mnemonic)                      sk_enc (generated in TEE)
    pk_a (on-chain)                           pk_enc (in attestation doc)

         ─── pk_enc (via attestation) ──────►
         ◄── pk_a (via FitRes.metrics) ──────

    shared = ECDH(sk_a, pk_enc)               shared = ECDH(sk_enc, pk_a)
         │                                          │
         │  (both sides get same shared secret)     │
         ▼                                          ▼
    sym_key = HKDF-SHA256(shared,             sym_key = HKDF-SHA256(shared,
              info="rizemind-tee-v1")                   info="rizemind-tee-v1")
         │                                          │
         ▼                                          ▼
    ciphertext = AES-256-GCM-Enc(             plaintext = AES-256-GCM-Dec(
                   sym_key, params)                      sym_key, ciphertext)
```

### Attestation Verification Flow

```
    ┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
    │  AWS Nitro      │     │  Nitro Enclave       │     │  Client          │
    │  Hypervisor     │     │                      │     │  (Trainer/Eval)  │
    └────────┬────────┘     └───────────┬──────────┘     └─────────┬────────┘
             │                          │                          │
             │    NSM.get_attestation   │                          │
             │◄─────────────────────────│                          │
             │   (includes pk_enclave)  │                          │
             │                          │                          │
             │  Sign with Nitro root CA │                          │
             │  (COSE_Sign1 / ES384)    │                          │
             │─────────────────────────►│                          │
             │   attestation_document:  │                          │
             │   • PCR0 (code hash)     │  send via Flower config  │
             │   • PCR1 (kernel hash)   │─────────────────────────►│
             │   • pk_enclave           │                          │
             │   • cert_chain           │                          │  Verify:
             │   • signature            │                          │  1. Decode CBOR
             │                          │                          │  2. Check COSE sig
             │                          │                          │  3. Validate cert
             │                          │                          │     chain → AWS
             │                          │                          │     Nitro root CA
             │                          │                          │  4. PCR0 matches
             │                          │                          │     expected code
             │                          │                          │  5. pk_enclave is
             │                          │                          │     bound in doc
             │                          │                          │
             │                          │                          │  ✓ Trust pk_enclave
             │                          │                          │    for ECDH
    ─────────┴──────────────────────────┴──────────────────────────┴──────────
```

### Strategy Decorator Composition

```
    Outermost (first to receive calls)
    ┌─────────────────────────────────────────────────────┐
    │  EthAccountStrategy                                  │
    │  • Authenticates clients (EIP-712 signatures)        │
    │  • Filters unauthorized trainers → failures          │
    │  ┌─────────────────────────────────────────────────┐ │
    │  │  ShapleyValueStrategy                           │ │
    │  │  • Stores FitRes per trainer                    │ │
    │  │  • Creates coalitions for contribution calc     │ │
    │  │  ┌─────────────────────────────────────────────┐│ │
    │  │  │  TEEAggregationStrategy                     ││ │
    │  │  │  • configure_fit: attach pk_enclave         ││ │
    │  │  │  • aggregate_fit: collect encrypted,        ││ │
    │  │  │    forward to TEE, return aggregated        ││ │
    │  │  │  ┌─────────────────────────────────────────┐││ │
    │  │  │  │  FedAvg (base strategy)                 │││ │
    │  │  │  │  • Runs inside the TEE enclave          │││ │
    │  │  │  │  • Never sees raw params on the server  │││ │
    │  │  │  └─────────────────────────────────────────┘││ │
    │  │  └─────────────────────────────────────────────┘│ │
    │  └─────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────┘
    Innermost (actual aggregation logic)
```

### Client-Side Mod Chain

```
    Incoming TRAIN message (from server)
              │
              ▼
    ┌─────────────────────────┐
    │  model_notary_mod       │   Verifies server's model signature
    │  (runs first)           │   Signs outgoing model with trainer's key
    └────────────┬────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │  tee_encryption_mod     │   Reads pk_enclave from config
    │  (runs second)          │   ECDH → AES-GCM encrypt parameters
    └────────────┬────────────┘   Puts ciphertext in FitRes.metrics
              │
              ▼
    Outgoing FitRes (to server)
    • metrics["tee_encrypted_params"] = ciphertext
    • metrics["tee_nonce"]            = nonce
    • metrics["tee_client_pubkey"]    = pk_trainer
    • metrics["notary_signature"]     = EIP-712 sig
```

---

## Sequence Diagram — Full System Flow

```
 Blockchain       Trainer             Aggregator         TEE Enclave        Evaluator
     │               │                    │                   │                  │
     │               │                    │                   │                  │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │  PHASE 0: ONE-TIME REGISTRATION (each entity registers on-chain)          │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │               │                    │                   │                  │
     │◄──register────│                    │                   │                  │
     │  mnemonic → HD derive              │                   │                  │
     │  → sk_trainer (secp256k1)          │                   │                  │
     │  → pk_trainer on-chain             │                   │                  │
     │               │                    │                   │                  │
     │◄─────────────────register──────────│                   │                  │
     │               │  mnemonic → HD derive                  │                  │
     │               │  → sk_aggregator (secp256k1)           │                  │
     │               │  → pk_aggregator on-chain              │                  │
     │               │                    │                   │                  │
     │◄──────────────────────────────────────────────────────────register────────│
     │               │                    │                   │  mnemonic → HD   │
     │               │                    │                   │  → sk_eval       │
     │               │                    │                   │  → pk_eval       │
     │               │                    │                   │    on-chain      │
     │               │                    │                   │                  │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │  PHASE 1: ENCLAVE BOOT (once per session)              │                  │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │               │                    │                   │                  │
     │               │                    │──nitro-cli run──► │                  │
     │               │                    │                   │                  │
     │               │                    │                   │─┐ generate       │
     │               │                    │                   │ │ ECDH keypair   │
     │               │                    │                   │◄┘ (sk_enc,       │
     │               │                    │                   │    pk_enc)       │
     │               │                    │                   │                  │
     │               │                    │                   │─┐ request NSM    │
     │               │                    │                   │ │ attestation    │
     │               │                    │                   │◄┘ (embeds        │
     │               │                    │                   │    pk_enc)       │
     │               │                    │                   │                  │
     │               │                    │◄─vsock:5000───────│                  │
     │               │                    │  attestation_doc: │                  │
     │               │                    │  • pk_enc         │                  │
     │               │                    │  • PCR0 (code)    │                  │
     │               │                    │  • cert chain     │                  │
     │               │                    │  • COSE signature │                  │
     │               │                    │                   │                  │
     │               │                    │─┐ verify:         │                  │
     │               │                    │ │ COSE_Sign1 sig  │                  │
     │               │                    │ │ cert → AWS CA   │                  │
     │               │                    │◄┘ PCR0 match      │                  │
     │               │                    │                   │                  │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │  PHASE 2: TRAINING ROUND (repeats each round)          │                  │
 ════╪═══════════════╪════════════════════╪═══════════════════╪══════════════════╪════
     │               │                    │                   │                  │
     │               │                    │─┐ configure_fit() │                  │
     │               │                    │ │ attach pk_enc + │                  │
     │               │                    │◄┘ attestation to  │                  │
     │               │                    │   FitIns.config   │                  │
     │               │                    │                   │                  │
     │               │◄───FitIns──────────│                   │                  │
     │               │  config:           │                   │                  │
     │               │  • global_params   │                   │                  │
     │               │  • pk_enc          │                   │                  │
     │               │  • attestation_doc │                   │                  │
     │               │                    │                   │                  │
     │               │─┐ (optional)       │                   │                  │
     │               │ │ verify           │                   │                  │
     │               │◄┘ attestation      │                   │                  │
     │               │                    │                   │                  │
     │               │─┐ LOCAL TRAINING   │                   │                  │
     │               │ │ train model on   │                   │                  │
     │               │◄┘ private data     │                   │                  │
     │               │                    │                   │                  │
     │               │─┐ ECDH ENCRYPT     │                   │                  │
     │               │ │                  │                   │                  │
     │               │ │ sk = ec_key_from_account(wallet)     │                  │
     │               │ │ shared = ECDH(sk_trainer, pk_enc)    │                  │
     │               │ │ sym_key = HKDF-SHA256(shared)        │                  │
     │               │ │ (ct, nonce) = AES-GCM(sym_key,       │                  │
     │               │◄┘                        params)       │                  │
     │               │                    │                   │                  │
     │               │────FitRes─────────►│                   │                  │
     │               │  metrics:          │                   │                  │
     │               │  • ciphertext      │                   │                  │
     │               │  • nonce           │                   │                  │
     │               │  • pk_trainer      │                   │                  │
     │               │  • num_examples    │                   │                  │
     │               │                    │                   │                  │
     │          (× N trainers in parallel)│                   │                  │
     │               │                    │                   │                  │
     │               │                    │─┐ aggregate_fit() │                  │
     │               │                    │ │ collect all     │                  │
     │               │                    │◄┘ encrypted       │                  │
     │               │                    │   FitRes          │                  │
     │               │                    │                   │                  │
     │               │                    │───vsock AGGREGATE─►│                  │
     │               │                    │  payload:          │                  │
     │               │                    │  • [(ct_i, nonce_i,│                  │
     │               │                    │     pk_i, n_i)...] │                  │
     │               │                    │                    │                  │
     │               │                    │                    │─┐ FOR EACH       │
     │               │                    │                    │ │ TRAINER i:     │
     │               │                    │                    │ │                │
     │               │                    │                    │ │ shared_i =     │
     │               │                    │                    │ │  ECDH(sk_enc,  │
     │               │                    │                    │ │       pk_i)    │
     │               │                    │                    │ │ key_i =        │
     │               │                    │                    │ │  HKDF(shared_i)│
     │               │                    │                    │ │ params_i =     │
     │               │                    │                    │ │  AES-Dec(key_i,│
     │               │                    │                    │ │   nonce_i,     │
     │               │                    │                    │◄┘   ct_i)       │
     │               │                    │                    │                  │
     │               │                    │                    │─┐ AGGREGATE      │
     │               │                    │                    │ │ FedAvg.        │
     │               │                    │                    │ │ aggregate_fit( │
     │               │                    │                    │ │  all decrypted │
     │               │                    │                    │ │  params,       │
     │               │                    │                    │◄┘  weights)      │
     │               │                    │                    │                  │
     │               │                    │◄──vsock────────────│                  │
     │               │                    │  aggregated_params │                  │
     │               │                    │                    │                  │
 ════╪═══════════════╪════════════════════╪════════════════════╪══════════════════╪════
     │  PHASE 3: EVALUATION (same round, after aggregation)    │                  │
 ════╪═══════════════╪════════════════════╪════════════════════╪══════════════════╪════
     │               │                    │                    │                  │
     │               │                    │─┐ configure_eval() │                  │
     │               │                    │ │                   │                  │
     │               │                    │ │ FOR EACH EVAL j:  │                  │
     │               │                    │ │ pk_j = lookup     │                  │
     │◄──────────────────────────────read─┘ │  on-chain         │                  │
     │──pk_eval_j──────────────────────────►│                   │                  │
     │               │                    │                     │                  │
     │               │                    │───vsock ENCRYPT────►│                  │
     │               │                    │  (agg_params, pk_j) │                  │
     │               │                    │                     │                  │
     │               │                    │                     │─┐ ECDH ENCRYPT  │
     │               │                    │                     │ │ shared_j =    │
     │               │                    │                     │ │  ECDH(sk_enc, │
     │               │                    │                     │ │       pk_j)   │
     │               │                    │                     │ │ key_j =       │
     │               │                    │                     │ │  HKDF(shared) │
     │               │                    │                     │ │ ct_j =        │
     │               │                    │                     │ │  AES-Enc(     │
     │               │                    │                     │ │   key_j,      │
     │               │                    │                     │◄┘   agg_params) │
     │               │                    │                     │                  │
     │               │                    │◄──vsock─────────────│                  │
     │               │                    │  (ct_j, nonce_j)    │                  │
     │               │                    │                     │                  │
     │               │                    │────EvaluateIns─────────────────────────►│
     │               │                    │  config:            │                  │
     │               │                    │  • pk_enc           │                  │
     │               │                    │  • attestation_doc  │                  │
     │               │                    │  parameters:        │                  │
     │               │                    │  • encrypted model  │                  │
     │               │                    │  • nonce            │                  │
     │               │                    │                     │                  │
     │               │                    │                     │  ┌─ ECDH DECRYPT│
     │               │                    │                     │  │ sk =         │
     │               │                    │                     │  │  ec_key_from │
     │               │                    │                     │  │  _account(   │
     │               │                    │                     │  │   wallet)    │
     │               │                    │                     │  │ shared =     │
     │               │                    │                     │  │  ECDH(sk_eval│
     │               │                    │                     │  │      pk_enc) │
     │               │                    │                     │  │ key =        │
     │               │                    │                     │  │  HKDF(shared)│
     │               │                    │                     │  │ params =     │
     │               │                    │                     │  │  AES-Dec(key,│
     │               │                    │                     │  └─  ct_j)     │
     │               │                    │                     │                  │
     │               │                    │                     │      ┌─ EVALUATE │
     │               │                    │                     │      │ run model │
     │               │                    │                     │      │ on test   │
     │               │                    │                     │      └─ dataset  │
     │               │                    │                     │                  │
     │               │                    │◄───EvaluateRes────────────────────────│
     │               │                    │  loss: 0.42         │                  │
     │               │                    │  metrics:           │                  │
     │               │                    │  • accuracy: 0.91   │                  │
     │               │                    │  (plaintext scores  │                  │
     │               │                    │   — no encryption   │                  │
     │               │                    │   needed on return) │                  │
     │               │                    │                     │                  │
 ════╪═══════════════╪════════════════════╪═════════════════════╪══════════════════╪════
     │  PHASE 4: ON-CHAIN RECORDING                             │                  │
 ════╪═══════════════╪════════════════════╪═════════════════════╪══════════════════╪════
     │               │                    │                     │                  │
     │               │                    │─┐ sign tx with      │                  │
     │               │                    │ │ sk_aggregator     │                  │
     │               │                    │◄┘ (wallet key)      │                  │
     │               │                    │                     │                  │
     │◄──────────────────submit tx────────│                     │                  │
     │  record:      │                    │                     │                  │
     │  • round #    │                    │                     │                  │
     │  • agg hash   │                    │                     │                  │
     │  • eval scores│                    │                     │                  │
     │  • Shapley    │                    │                     │                  │
     │    values     │                    │                     │                  │
     │               │                    │                     │                  │
     │               │                    │              ROUND COMPLETE             │
     │               │                    │         (loop back to Phase 2)          │
```

---

## Two ECDH Channels

| Channel       | From           | To          | What's encrypted         | When                 | Status                  |
| ------------- | -------------- | ----------- | ------------------------ | -------------------- | ----------------------- |
| 1. Training   | Trainers       | TEE Enclave | Individual model updates | `aggregate_fit`      | **Implemented**         |
| 2. Evaluation | TEE/Aggregator | Evaluators  | Aggregated model         | `configure_evaluate` | **Not yet implemented** |

Channel 1 is the critical path — it protects individual training data from the aggregator operator. Channel 2 protects the aggregated model during distribution to evaluators. Evaluators return only scores (loss, metrics) — not model weights — so no encryption is needed on the return path.

---

## What Was Implemented

### 1. `tee/crypto.py` — Cryptographic Primitives

**What:**

- `generate_ecdh_keypair()` — generates a secp256k1 key pair
- `serialize_public_key()` / `deserialize_public_key()` — X9.62 uncompressed format (65 bytes)
- `derive_shared_secret(private, peer_public)` — raw ECDH exchange
- `derive_symmetric_key(shared_secret)` — HKDF-SHA256 → 32-byte AES key with domain separation (`rizemind-tee-v1`)
- `aes_gcm_encrypt(key, plaintext)` → `(ciphertext, nonce)` with authenticated additional data (`model-params`)
- `aes_gcm_decrypt(key, nonce, ciphertext)` → `plaintext`

### 2. `tee/params.py` — Serialization

**Why:** Flower's `Parameters` object (list of tensor byte buffers + type string) needs to be packed into a single byte buffer for encryption. We also need `num_examples` for weighted FedAvg.

**What:**

- `serialize_parameters()` / `deserialize_parameters()` — compact binary format with length prefixes
- `serialize_fit_res_for_enclave()` / `deserialize_fit_res_for_enclave()` — adds `num_examples` header

### 3. `tee/enclave.py` — TEE Abstraction

**Why:** Decouples the strategy and client mod from any specific TEE platform. The same code works with the mock (for testing) and Nitro (for production).

**What:**

- `AttestationReport` — frozen dataclass holding the enclave's public key, platform attestation document, PCR values, and timestamp
- `TEEEnclave` — abstract base class with lifecycle methods: `initialize()`, `get_attestation_report()`, `get_public_key()`, `aggregate(encrypted_updates, num_examples, server_round)`, `destroy()`

### 4. `tee/attestation.py` — Attestation Verification

**Why:** Before encrypting data for the enclave, clients need to verify that the enclave's public key genuinely belongs to a real TEE running the expected code — not a rogue process pretending to be one.

**What:**

- `AttestationVerifier` — abstract base with `verify(report) -> bool`
- `MockAttestationVerifier` — always returns True (for testing)
- `NitroAttestationVerifier` — decodes CBOR/COSE_Sign1, validates certificate chain against AWS Nitro root CA, checks PCR values

### 5. `tee/vsock.py` — Vsock Communication

**Why:** Nitro Enclaves communicate exclusively over vsock (virtual socket) — no network, no filesystem. Model parameters can be many megabytes, so we need reliable length-framed messaging.

**What:**

- `vsock_send(sock, data)` — sends `[8-byte length header][payload]`
- `vsock_recv(sock)` — reads length, then exactly that many bytes
- Handles partial reads and connection drops

### 6. `tee/nitro/enclave_server.py` — Code That Runs Inside the Enclave

**Why:** This is the trusted code. It generates the ECDH key pair inside the enclave (the private key never leaves), requests an NSM attestation document embedding the public key, and performs the actual decryption + aggregation.

**What:**

- Generates ECDH keypair on startup
- Requests attestation from NSM API with public key embedded
- Listens on vsock port 5000
- `GET_ATTESTATION` command: returns public key + attestation document
- `AGGREGATE` command: for each trainer → ECDH derive shared secret → AES-GCM decrypt → reconstruct Flower `FitRes` → call `FedAvg().aggregate_fit()` → return serialized aggregated `Parameters`

### 7. `tee/nitro/nitro_enclave.py` — Parent-Side Enclave Management

**Why:** The parent EC2 instance needs to start/stop the enclave and communicate with it. This implements the `TEEEnclave` interface.

**What:**

- `initialize()` — runs `nitro-cli run-enclave`, gets CID, connects over vsock to fetch attestation
- `aggregate()` — builds binary payload, sends over vsock, receives aggregated result
- `destroy()` — runs `nitro-cli terminate-enclave`

### 8. `tee/mock_enclave.py` — Software Mock

**Why:** Development and testing without AWS/Nitro hardware. Runs the exact same ECDH + decrypt + FedAvg logic in-process.

### 9. `tee/tee_strategy.py` — Strategy Decorator

**Why:** Follows the existing decorator pattern (`EthAccountStrategy`, `MetricStorageStrategy`) so TEE integration composes cleanly with auth and Shapley strategies.

**What:**

- `TEEAggregationStrategy(Strategy)` — wraps any base strategy
- `configure_fit()` — attaches enclave's ECDH public key + attestation to each client's config
- `aggregate_fit()` — extracts encrypted updates from `FitRes.metrics`, sends to enclave, returns decrypted aggregated result
- Lazy initialization: enclave starts on first `configure_fit` call
- Delegates `configure_evaluate`, `aggregate_evaluate`, `evaluate` to the wrapped strategy (Channel 2 will be added here)

### 10. `tee/tee_client_mod.py` — Client-Side Encryption Mod

**Why:** Trainers need to encrypt their model updates before sending. This is a Flower "mod" (middleware) following the same pattern as `model_notary_mod`.

**What:**

- Intercepts TRAIN reply messages
- Reads TEE public key from `FitIns.config`
- Generates an **ephemeral** ECDH keypair (new each round → forward secrecy)
- Derives shared secret with TEE → HKDF → AES-256-GCM encrypts `FitRes.parameters`
- Stores `(ciphertext, nonce, client_public_key)` in `FitRes.metrics`

### 11. `Dockerfile.enclave`

Builds the Docker image that `nitro-cli build-enclave` converts into an EIF (Enclave Image File). Contains only the minimal dependencies needed inside the enclave: `flwr`, `numpy`, `cryptography`.

### 12. Dependencies

Added `tee` dependency group to `pyproject.toml`: `cbor2` and `cose` for Nitro attestation verification.

---

## What's NOT Yet Implemented

### Evaluator ECDH Channel (Channel 2: Aggregator → Evaluators)

Currently `configure_evaluate` passes through to the base strategy without encryption. To complete:

- In `configure_evaluate`: the TEE (or aggregator on behalf of TEE) encrypts the aggregated model with each evaluator's ECDH public key before sending via `EvaluateIns.parameters`
- Evaluators decrypt using their ephemeral key + TEE public key
- A `tee_evaluation_mod` (client-side) would handle the decryption for evaluators
- Evaluators return only scores (loss, metrics), not model weights — so the return path does not need encryption

This can be added by extending `TEEAggregationStrategy.configure_evaluate()` and creating a companion evaluator-side client mod, reusing the same `crypto.py` primitives.

---

## File Summary

| File                          | Purpose                                |
| ----------------------------- | -------------------------------------- |
| `tee/__init__.py`             | Package exports                        |
| `tee/crypto.py`               | ECDH + HKDF + AES-GCM primitives       |
| `tee/params.py`               | Flower Parameters serialization        |
| `tee/enclave.py`              | `TEEEnclave` ABC + `AttestationReport` |
| `tee/attestation.py`          | Attestation verifiers (Mock, Nitro)    |
| `tee/vsock.py`                | Length-framed vsock messaging          |
| `tee/mock_enclave.py`         | In-process mock for testing            |
| `tee/tee_strategy.py`         | `TEEAggregationStrategy` decorator     |
| `tee/tee_client_mod.py`       | Client-side encryption mod             |
| `tee/nitro/__init__.py`       | Nitro subpackage                       |
| `tee/nitro/enclave_server.py` | Runs inside Nitro Enclave              |
| `tee/nitro/nitro_enclave.py`  | Parent-side enclave management         |
| `tee/Dockerfile.enclave`      | Docker → EIF build                     |

## Test Coverage

27 tests covering:

- ECDH key exchange round-trips (shared secret agreement)
- AES-GCM encrypt/decrypt (including wrong-key and tamper detection)
- Parameters serialization round-trips
- Mock enclave: full pipeline with FedAvg correctness verification
- Strategy integration: encrypted updates through the decorator
