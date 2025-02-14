import pytorch_lightning as pl
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.trainer.states import TrainerFn
from opacus import PrivacyEngine
from typing import Any, Callable, Optional
import torch
from torch.utils.data import DataLoader


class DPStrategy(SingleDeviceStrategy):
    def __init__(
        self,
        train_dataloader: DataLoader,
        after_setup,
        device: str = "cpu",
        delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        poisson_sampling: bool = False,
    ):
        super().__init__(device)
        self.train_dataloader = train_dataloader
        self.after_setup = after_setup
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        self.poisson_sampling = poisson_sampling
        self.privacy_engine = PrivacyEngine()

    def setup(self, trainer: pl.Trainer) -> None:
        super().setup(trainer)

        optimizers = self.model.configure_optimizers()

        # Assert that only one optimizer is returned
        if isinstance(optimizers, torch.optim.Optimizer):
            optimizer = optimizers
        elif (
            isinstance(optimizers, (list, tuple))
            and len(optimizers) == 1
            and isinstance(optimizers[0], torch.optim.Optimizer)
        ):
            optimizer = optimizers[0]
        else:
            raise ValueError(
                "Expected `configure_optimizers` to return a single optimizer instance."
            )

        all_params = [p for group in optimizer.param_groups for p in group["params"]]
        max_grad_norm = [self.max_grad_norm] * len(all_params)

        # Apply PrivacyEngine to the model and the optimizer
        self.model, optimizer, dataloader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=dataloader,  # noqa: F821
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=self.poisson_sampling,
            clipping="per_layer",
        )
        trainer.optimizers = [optimizers]
        self.after_setup

    # TODO: fix error num_row was 18k now 6k
    def process_dataloader(self, dataloader: Any) -> Any:
        if self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            return self.privacy_engine._prepare_data_loader(
                dataloader, distributed=False, poisson_sampling=self.poisson_sampling
            )
        return dataloader

    def optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        closure: Callable[[], Any],
        model: Optional[pl.LightningModule] = None,
        **kwargs: Any,
    ) -> Any:
        optimizer.step(closure=closure, **kwargs)
        optimizer.zero_grad()  # likely not required
