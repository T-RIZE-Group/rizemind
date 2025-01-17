from flwr.server.strategy import Strategy
from flwr.common.typing import FitRes
from rize_dml.contracts.deploy.model_registry_v1 import TrainerGroup
from rize_dml.authentication.signature import recover_model_signer

class EthAccountStrategy(Strategy):

  strat: Strategy
  group: TrainerGroup

  def __init__(self, 
    strat: Strategy,
    group: TrainerGroup
  ):
      super().__init__()
      self.strat = strat
      self.group = group

  def initialize_parameters(self, client_manager):
      return self.strat.initialize_parameters(client_manager)

  def configure_fit(self, server_round, parameters, client_manager):
      return self.strat.configure_fit(server_round, parameters, client_manager)
      
  def aggregate_fit(self, server_round, results, failures):
      whitelisted = []
      for client, res in results:
        signer = self._recover_signer(res)
        if self.group.validate_signer(signer):
            whitelisted.append((client, res))

      return self.strat.aggregate_fit(server_round, whitelisted, failures)

  def _recover_signer(self, res: FitRes):
      vrs = (res.metrics.get("v"), res.metrics.get("r"), res.metrics.get("s"))
      signer = recover_model_signer(res.parameters, self.group.chain_id, self.group.contract.address, self.group.name(), 0, vrs)
      return signer

  def configure_evaluate(self, server_round, parameters, client_manager):
      return self.strat.configure_evaluate(server_round, parameters, client_manager)

  def aggregate_evaluate(self, server_round, results, failures):
      return self.strat.aggregate_evaluate(server_round, results, failures)

  def evaluate(self, server_round, parameters):
      return self.strat.evaluate(server_round, parameters)