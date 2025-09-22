from hexbytes import HexBytes
from web3 import Web3

IDLE_PHASE = "IDLE"
IDLE_PHASE_ID = Web3.keccak(text=IDLE_PHASE)

TRAINING_PHASE = "TRAINING"
TRAINING_PHASE_ID = Web3.keccak(text=TRAINING_PHASE)

EVALUATOR_REGISTRATION_PHASE = "EVALUATOR_REGISTRATION"
EVALUATOR_REGISTRATION_PHASE_ID = Web3.keccak(text=EVALUATOR_REGISTRATION_PHASE)

EVALUATION_PHASE = "EVALUATION"
EVALUATION_PHASE_ID = Web3.keccak(text=EVALUATION_PHASE)


def get_phase_name(phase_id: HexBytes) -> str:
    """Get human-readable name for a phase ID."""
    if phase_id == IDLE_PHASE_ID:
        return "IDLE"
    elif phase_id == TRAINING_PHASE_ID:
        return "TRAINING"
    elif phase_id == EVALUATOR_REGISTRATION_PHASE_ID:
        return "EVALUATOR_REGISTRATION"
    elif phase_id == EVALUATION_PHASE_ID:
        return "EVALUATION"
    else:
        return f"UNKNOWN_PHASE_{phase_id.hex()}"
