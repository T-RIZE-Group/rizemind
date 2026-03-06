"""TEE aggregation strategy decorator.

``TEEAggregationStrategy`` wraps any Flower ``Strategy`` and routes the
``aggregate_fit`` call through a TEE enclave.  It follows the same
decorator pattern as ``EthAccountStrategy`` in
``rizemind.authentication.eth_account_strategy``.

Server-side usage::

    base = FedAvg(...)
    enclave = MockTEEEnclave()            # or NitroTEEEnclave(eif_path)
    verifier = MockAttestationVerifier()  # or NitroAttestationVerifier(...)
    strategy = TEEAggregationStrategy(base, enclave, verifier)

    # Can be further wrapped with auth:
    strategy = EthAccountStrategy(strategy, swarm, account)
"""

import json
import logging

from flwr.common import log
from flwr.common.typing import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from rizemind.exception import RizemindException
from rizemind.tee.attestation import AttestationVerifier
from rizemind.tee.enclave import AttestationReport, TEEEnclave
from rizemind.tee.params import deserialize_parameters

# Keys used in Flower's config/metrics dicts to carry TEE data
TEE_PUBLIC_KEY_CONFIG = "tee_public_key"
TEE_ATTESTATION_CONFIG = "tee_attestation"
TEE_ENCRYPTED_PARAMS_METRIC = "tee_encrypted_params"
TEE_NONCE_METRIC = "tee_nonce"
TEE_CLIENT_PUBKEY_METRIC = "tee_client_pubkey"


class TEEAttestationFailedException(RizemindException):
    def __init__(self) -> None:
        super().__init__(
            code="tee_attestation_failed",
            message="TEE attestation verification failed",
        )


class TEEAggregationStrategy(Strategy):
    """Strategy decorator that routes model aggregation through a TEE.

    In ``configure_fit``, attaches the TEE's public key and attestation to
    each client's config so they can encrypt their updates.

    In ``aggregate_fit``, collects the encrypted updates from
    ``FitRes.metrics`` and sends them to the enclave for decryption and
    Flower-native aggregation (FedAvg).
    """

    strat: Strategy
    enclave: TEEEnclave
    verifier: AttestationVerifier
    _attestation: AttestationReport | None

    def __init__(
        self,
        strat: Strategy,
        enclave: TEEEnclave,
        verifier: AttestationVerifier,
    ) -> None:
        super().__init__()
        self.strat = strat
        self.enclave = enclave
        self.verifier = verifier
        self._attestation = None

    def _ensure_enclave(self) -> None:
        """Lazy-initialize the enclave and verify its attestation."""
        if self._attestation is None:
            self.enclave.initialize()
            self._attestation = self.enclave.get_attestation_report()
            if not self.verifier.verify(self._attestation):
                raise TEEAttestationFailedException()
            log(
                logging.INFO,
                "TEE enclave initialized and attested (platform=%s)",
                self._attestation.platform,
            )

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        self._ensure_enclave()
        return self.strat.initialize_parameters(client_manager)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Attach TEE public key and attestation to each client's fit config."""
        self._ensure_enclave()
        client_instructions = self.strat.configure_fit(
            server_round, parameters, client_manager
        )
        tee_pubkey = self.enclave.get_public_key()
        attestation_bytes = self._serialize_attestation(self._attestation)

        for _, fit_ins in client_instructions:
            fit_ins.config[TEE_PUBLIC_KEY_CONFIG] = tee_pubkey
            fit_ins.config[TEE_ATTESTATION_CONFIG] = attestation_bytes

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Collect encrypted updates and delegate aggregation to TEE."""
        encrypted_updates: list[tuple[bytes, bytes, bytes]] = []
        num_examples: list[int] = []

        for client, res in results:
            metrics = res.metrics
            if (
                TEE_ENCRYPTED_PARAMS_METRIC in metrics
                and TEE_NONCE_METRIC in metrics
                and TEE_CLIENT_PUBKEY_METRIC in metrics
            ):
                encrypted_updates.append((
                    metrics[TEE_ENCRYPTED_PARAMS_METRIC],
                    metrics[TEE_NONCE_METRIC],
                    metrics[TEE_CLIENT_PUBKEY_METRIC],
                ))
                num_examples.append(res.num_examples)
            else:
                log(
                    logging.WARNING,
                    "Client result missing TEE encryption metadata, skipping",
                )
                failures.append(
                    RizemindException(
                        code="missing_tee_metadata",
                        message="Client did not encrypt model update for TEE",
                    )
                )

        if not encrypted_updates:
            log(logging.WARNING, "No encrypted updates received for TEE aggregation")
            return None, {}

        # Delegate to TEE enclave
        aggregated_bytes = self.enclave.aggregate(
            encrypted_updates=encrypted_updates,
            num_examples=num_examples,
            server_round=server_round,
        )
        aggregated_params = deserialize_parameters(aggregated_bytes)

        log(
            logging.INFO,
            "TEE aggregated %d updates in round %d",
            len(encrypted_updates),
            server_round,
        )
        return aggregated_params, {"tee_aggregated_count": len(encrypted_updates)}

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        return self.strat.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        return self.strat.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> tuple[float, dict[str, Scalar]] | None:
        return self.strat.evaluate(server_round, parameters)

    @staticmethod
    def _serialize_attestation(report: AttestationReport | None) -> bytes:
        """Serialize attestation report for transmission via Flower config."""
        if report is None:
            return b""
        return json.dumps({
            "enclave_public_key": report.enclave_public_key.hex(),
            "document": report.document.hex(),
            "platform": report.platform,
            "timestamp": report.timestamp,
            "pcrs": {str(k): v.hex() for k, v in report.pcrs.items()},
        }).encode("utf-8")
