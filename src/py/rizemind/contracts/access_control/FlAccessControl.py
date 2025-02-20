from web3.contract import Contract


class FlAccessControl:
    fl_contract: Contract

    def __init__(self, model: Contract):
        self.fl_contract = model

    def is_trainer(self, address: str) -> bool:
        return self.fl_contract.functions.isTrainer(address).call()

    def is_aggregator(self, address: str) -> bool:
        return self.fl_contract.functions.isAggregator(address).call()
