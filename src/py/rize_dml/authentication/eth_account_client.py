from typing import Tuple
from flwr.client import Client
from flwr.common import FitRes, FitIns, EvaluateIns
from eth_account import Account
from eth_account.datastructures import SignedMessage
from rize_dml.authentication.signature import sign_parameters_model


class SigningClient:

  client: Client
  account: Account

  def __init__(
    self, 
    client: Client,
    account: Account
  ):
    self.client = client
    self.account = account

  def __getattr__(self, name):
    return getattr(self.client, name)

  def fit(self, ins: FitIns):
    # Call the original fit method on the proxied Client
    results: FitRes = self.client.fit(ins)
    signature = self._sign(results)
    results.metrics = results.metrics | signature
    return results
  
  def _sign(self, res: FitRes) -> SignedMessage:
    
    signature = sign_parameters_model(
      self.account,
      res.parameters,
      1,
      "0xCcCCccccCCCCcCCCCCCcCcCccCcCCCcCcccccccC",
      "test_model",
      0    
    )
    return {
      "r": signature.r.to_bytes(32, byteorder='big'),
      "s": signature.s.to_bytes(32, byteorder='big'),
      "v": signature.v.to_bytes(1, byteorder='big')
    }
  
  def evaluate(self, ins: EvaluateIns):
    return self.client.evaluate(ins)
