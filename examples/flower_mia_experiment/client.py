import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # Local training
        self.net.train()
        for epoch in range(1): # 1 local epoch per round for simplicity
            for images, labels in self.trainloader:
                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.net.eval()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in self.valloader:
                outputs = self.net(images)
                loss += self.criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                
        accuracy = correct / len(self.valloader.dataset)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}
