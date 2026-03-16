import io
import random
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class RandomJPEGCompression:
    def __init__(self, quality=50, p=0.5):
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    
class AddGaussianNoise:
    def __init__(self, std=0.3, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        x = x + noise
        return x.clamp(0.0, 1.0)
    
class FixedBrightness:
    def __init__(self, factor=0.5, p=0.5):
        self.factor = factor
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        return TF.adjust_brightness(img, self.factor)

class RajarshiDataset(Dataset):
    def __init__(self, dataset, transform=None, return_source=False):
        self.dataset = dataset
        self.transform = transform
        self.return_source = return_source

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["Image"]
        label_a = row["Label_A"]   # 0 = real, 1 = fake

        if self.transform is not None:
            image = self.transform(image)

        if self.return_source:
            label_b = row["Label_B"]
            return image, label_a, label_b

        return image, label_a


def build_transforms(image_size: int):
    # Reference: https://arxiv.org/pdf/2503.10718 -> 4.1.1 Data

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandomJPEGCompression(quality=50, p=0.2),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(5.0, 5.0)),
        FixedBrightness(factor=0.5, p=0.2),
        transforms.ToTensor(),
        AddGaussianNoise(std=0.3, p=0.2),
    ])


    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    return train_transform, eval_transform


def build_dataloaders(cfg):
    train_transform, eval_transform = build_transforms(cfg.image_size)

    train_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="train"
    )
    val_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="validation"
    )
    test_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="test"
    )

    train_dataset = RajarshiDataset(
        train_hf,
        transform=train_transform,
        return_source=False,
    )
    val_dataset = RajarshiDataset(
        val_hf,
        transform=eval_transform,
        return_source=True,
    )
    test_dataset = RajarshiDataset(
        test_hf,
        transform=eval_transform,
        return_source=True,
    )

    persistent = cfg.persistent_workers if cfg.num_workers > 0 else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, test_loader
