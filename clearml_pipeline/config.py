from pathlib import Path
from typing import Union

from pydantic_yaml import YamlModel


class AppConfig(YamlModel):
    # data
    dataset_path: Path
    dataset_id: str
    dataset_output_path: Path
    # model training
    batch_size: int
    test_batch_size: int
    epochs: int
    lr: float
    no_cuda: bool
    seed: int
    gamma: float
    # validation
    val_acc_threshold: float
    val_loss_threshold: float

    @classmethod
    def parse_raw(
        cls,
        filename: Union[str, Path] = str(Path(__file__).parent / "./config.yaml"),
        *args,
        **kwargs
    ):
        with open(filename, "r") as f:
            data = f.read()
        return super().parse_raw(data, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
