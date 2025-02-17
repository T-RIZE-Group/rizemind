import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
from pytorchlightning_example.dp_strategy import DPStrategy

from pytorchlightning_example.task import LitAutoEncoder, load_data


def main():
    """
    Using vanilla Lightning API to train/test
    """
    train, val, test = load_data(0, 1)
    model = LitAutoEncoder()

    trainer = pl.Trainer(
        max_epochs=10,
        strategy=DPStrategy(),
        devices=1,
        accelerator="cpu",
        callbacks=[Timer(duration="00:00:00:30")],
    )
    trainer.fit(model, train, val)
    trainer.test(model, test)


if __name__ == "__main__":
    main()
