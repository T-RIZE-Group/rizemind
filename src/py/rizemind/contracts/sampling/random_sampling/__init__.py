import os
from pathlib import Path

from hexbytes import HexBytes
from pydantic import Field, field_validator
from rizemind.contracts.abi import encode_with_selector
from rizemind.contracts.abi_helper import load_abi
from rizemind.contracts.sampling.selector_factory import SelectorConfig
from web3 import Web3

RANDOM_SAMPLING_SELECTOR_NAME = "random-sampling"

abi = load_abi(Path(os.path.dirname(__file__)) / "./abi.json")


class RandomSamplingSelectorConfig(SelectorConfig):
    name: str = RANDOM_SAMPLING_SELECTOR_NAME
    version: str = "1.0.0"
    ratio: float = Field(..., description="The ratio of samples to take")

    @field_validator("ratio")
    @classmethod
    def validate_ratio(cls, v: float) -> float:
        if not (0 <= v <= 1):
            raise ValueError("ratio must be between 0 and 1")
        return v

    def get_init_data(self, w3: Web3 = Web3()) -> HexBytes:
        target_ratio = w3.to_wei(self.ratio, "ether")
        return encode_with_selector("initialize(uint256)", ["uint256"], [target_ratio])
