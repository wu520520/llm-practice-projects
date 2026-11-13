from pathlib import Path

import torch
from attr import dataclass

@dataclass
class SftConfig:
    root_path: str = Path(__file__).parent.parent
    data_path: str = root_path / 'data'
    model_path: str = root_path / 'Qwen3-0.6B-Base'
    train_data_size: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 3e-5
    batch_size: int = 1
    log_path: str = root_path / 'logs'
    log_iter: int = 100
    warmup_ratio: float = 0.1
    save_path: str = root_path / 'checkpoints'




if __name__ == '__main__':
    print(SftConfig().data_path)