from flwr.server.strategy import Strategy
from flwr.common.typing import FitRes
from rize_dml.authentication.signature import recover_model_signer
from rize_dml.contracts.models.model_registry_v1 import ModelRegistryV1

class EthAccountStrategy(Strategy):

  strat: Strategy
  model: ModelRegistryV1

  def __init__(self, 
    strat: Strategy,
    model: ModelRegistryV1
  ):
      super().__init__()
      self.strat = strat
      self.model = model

  def initialize_parameters(self, client_manager):
      return self.strat.initialize_parameters(client_manager)

  def configure_fit(self, server_round, parameters, client_manager):
      return self.strat.configure_fit(server_round, parameters, client_manager)
      
  def aggregate_fit(self, server_round, results, failures):
      whitelisted = []
      for client, res in results:
        signer = self._recover_signer(res)
        if self.model.can_train(signer, server_round):
            whitelisted.append((client, res))

      return self.strat.aggregate_fit(server_round, whitelisted, failures)

  def _recover_signer(self, res: FitRes):
      vrs = (res.metrics.get("v"), res.metrics.get("r"), res.metrics.get("s"))
      signer = recover_model_signer(res.parameters, self.model.chain_id, self.model.contract.address, self.model.name(), 0, vrs)
      return signer

  def configure_evaluate(self, server_round, parameters, client_manager):
      return self.strat.configure_evaluate(server_round, parameters, client_manager)

  def aggregate_evaluate(self, server_round, results, failures):
      return self.strat.aggregate_evaluate(server_round, results, failures)

  def evaluate(self, server_round, parameters):
      return self.strat.evaluate(server_round, parameters)