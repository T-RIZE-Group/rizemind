import pytest
from eth_account import Account
from rizemind.contracts.models.model_factory_v1 import (
    ModelFactoryV1,
    ModelFactoryV1Config,
)
from web3 import Web3

mnemonic = "test test test test test test test test test test test junk"


def test_model_meta_get_latest_contribution():
    Account.enable_unaudited_hdwallet_features()
    aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelFactoryV1Config(
        name="test",
        ticker="tst",
        local_factory_deployment_path="forge/output/31337/ModelRegistryFactory.json",
    )
    model_factory = ModelFactoryV1(model_config)
    model = model_factory.deploy(aggregator, [aggregator.address], w3)
    trainer1 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/1'/0/0")
    trainer2 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/2'/0/0")
    trainer3 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/3'/0/0")
    model.distribute([(trainer1.address, 1.0), (trainer2.address, 2.0)])
    round_1 = {"model_score": 0.9, "n_trainers": 2, "total_contribution": 3}
    model.next_round(
        1, round_1["n_trainers"], round_1["n_trainers"], round_1["total_contribution"]
    )
    model.distribute([(trainer2.address, 3.0), (trainer3.address, 4.0)])

    trainer1_latest_contribution = model.get_latest_contribution(trainer1.address)
    assert trainer1_latest_contribution == 1.0

    trainer2_latest_contribution = model.get_latest_contribution(trainer2.address)
    assert trainer2_latest_contribution == 3.0

    trainer_3_latest_contribution = model.get_latest_contribution(trainer3.address)
    assert trainer_3_latest_contribution == 4.0

    trainer4 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/4'/0/0")
    trainer4_latest_contribution = model.get_latest_contribution(trainer4.address)
    assert trainer4_latest_contribution is None


def test_model_meta_get_round_at():
    Account.enable_unaudited_hdwallet_features()
    aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelFactoryV1Config(
        name="test",
        ticker="tst",
        local_factory_deployment_path="forge/output/31337/ModelRegistryFactory.json",
    )
    model_factory = ModelFactoryV1(model_config)
    model = model_factory.deploy(aggregator, [], w3)

    current_block: int = w3.eth.block_number
    summary = model.get_round_at(current_block)

    assert summary.round_id == 1, "First round should be #1"
    assert summary.finished is False, "No rounds have finished yet"
    assert summary.metrics is None, "No metrics until a round finishes"

    ## finish round 1

    round_1_model_score: float = 0.9
    round_1_n_trainers: int = 2
    round_1_total_contribution: int = 3

    tx_hash = model.next_round(
        1,
        round_1_n_trainers,
        round_1_model_score,
        round_1_total_contribution,
    )

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    round_1_block: int = receipt["blockNumber"]

    summary = model.get_round_at(round_1_block)

    assert summary.finished is True
    assert summary.round_id == 1
    assert summary.metrics is not None
    assert summary.metrics.n_trainers == round_1_n_trainers
    assert summary.metrics.model_score == pytest.approx(round_1_model_score)
    assert summary.metrics.total_contributions == round_1_total_contribution

    later_block: int = round_1_block + 1
    future_summary = model.get_round_at(later_block)

    assert future_summary.finished is False
    assert future_summary.metrics is None
    assert future_summary.round_id == 2

    round_2_score: float = 0.95
    round_2_trainers: int = 3
    round_2_total_contrib: int = 5

    tx_hash = model.next_round(
        2, round_2_trainers, round_2_score, round_2_total_contrib
    )
    # ---- exact-block query  → finished == True for round 2
    r2_summary = model.get_round_at(later_block)
    assert r2_summary.finished is True
    assert r2_summary.round_id == 2
    assert r2_summary.metrics is not None
    assert r2_summary.metrics.n_trainers == round_2_trainers
    assert r2_summary.metrics.model_score == pytest.approx(round_2_score)
    assert r2_summary.metrics.total_contributions == round_2_total_contrib


def test_model_meta_get_last_contributed_round_summary():
    Account.enable_unaudited_hdwallet_features()
    aggregator = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/0'/0/0")
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
    model_config = ModelFactoryV1Config(
        name="test",
        ticker="tst",
        local_factory_deployment_path="forge/output/31337/ModelRegistryFactory.json",
    )
    model_factory = ModelFactoryV1(model_config)
    model = model_factory.deploy(aggregator, [], w3)

    trainer1 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/1'/0/0")
    trainer2 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/2'/0/0")
    # Distribute initial contributions
    tx_hash = model.distribute([(trainer1.address, 1.0), (trainer2.address, 2.0)])

    # finish round 1
    round_1_model_score: float = 0.9
    round_1_n_trainers: int = 2
    round_1_total_contribution: int = 3

    model.next_round(
        1,
        round_1_n_trainers,
        round_1_model_score,
        round_1_total_contribution,
    )

    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    round_1_block: int = receipt["blockNumber"]

    summary = model.get_round_at(round_1_block)

    assert summary.finished is True
    assert summary.round_id == 1
    # Distribute additional contributions in round 2
    model.distribute([(trainer2.address, 3.0)])

    # Fetch the summary of the last contributed round for trainer 1 (should be round 1)
    trainer1_round_summary = model.get_last_contributed_round_summary(trainer1.address)

    assert trainer1_round_summary is not None, "Trainer 1 should have a round summary"
    assert trainer1_round_summary.round_id == 1, "Trainer 1 last contributed to round 1"
    assert trainer1_round_summary.finished is True, "not finished"

    trainer2_round_summary = model.get_last_contributed_round_summary(trainer2.address)

    assert trainer2_round_summary is not None, "Trainer 2 should have a round summary"
    assert trainer2_round_summary.round_id == 2, (
        "Trainer 2 last contribution is in unfinished Round 2"
    )

    trainer3 = Account.from_mnemonic(mnemonic, account_path="m/44'/60'/3'/0/0")
    trainer3_round_summary = model.get_last_contributed_round_summary(trainer3.address)
    assert trainer3_round_summary is None, (
        "Trainer 3 should not have a round summary yet"
    )
