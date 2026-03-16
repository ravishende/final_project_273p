from dataclasses import dataclass

@dataclass
class Config:
    # dataset / dataloader
    data_root: str = "data"
    image_size: int = 224
    batch_size: int = 128
    num_workers: int = 12
    pin_memory: bool = True
    persistent_workers: bool = True

    # model
    model_name: str = "rgb"   # "rgb", "fft", "real_artifact_net"
    num_classes: int = 2
    pretrained: bool = True
    dropout: float = 0.2
    hidden_dim: int = 256

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-4

    # experiment tracking
    project: str = "ai-image-detection"
    run_name: str = "resnet18-rgb"
    save_dir: str = "checkpoints"
    seed: int = 0

    # early stopping / training control
    max_epochs: int = 30
    min_epochs: int = 5
    patience: int = 5
    monitor_metric: str = "auc"   # "loss", "acc", "precision", "recall", "f1", "auc"
    monitor_mode: str = "max"     # "max" for auc/f1/acc, "min" for loss
    min_delta: float = 1e-4