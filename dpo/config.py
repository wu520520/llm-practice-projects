from pathlib import Path

import torch
from attr import dataclass

@dataclass
class DpoConfig:
    root_path: str = Path(__file__).parent.parent
    data_path: str = root_path / 'dpo' / 'data' / 'ultrafeedback_binarized'
    model_path: str = root_path / 'dpo' / 'Qwen3-0.6B-Base'
    train_data_size: int = 5000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    learning_rate: float = 3e-5
    batch_size: int = 4
    log_path: str = root_path / 'dpo'/ 'logs'
    log_iter: int = 100
    warmup_ratio: float = 0.1
    save_path: str = root_path / 'dpo' / 'checkpoints'
    beta:float = 0.1




if __name__ == '__main__':
    print(DpoConfig().save_path)