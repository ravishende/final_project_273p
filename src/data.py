import io
import random
import kagglehub
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from pathlib import Path
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

class HemgDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"]
        label = row["label"]
        label = int(1 - label) # need 1 = fake and 0 = real
        if self.transform:
            image = self.transform(image)
        return image, label

class CIFAKEDataset(Dataset):
    def __init__(
        self,
        split="train",
        transform=None,
        validation_ratio=0.2,
        seed=42,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be 'train', 'validation', or 'test'")
        self.transform = transform
        # get images
        root = Path(
            kagglehub.dataset_download(
                "birdy654/cifake-real-and-ai-generated-synthetic-images"
            )
        )
        selected_samples = None
        if split == "test":
            # get test
            base_samples = self._collect_samples(root, "test")
            selected_samples = base_samples
        else:
            # get train or validation
            # there is no pre-defined validation set --> get a reproducible partition from training set
            base_samples = self._collect_samples(root, "train")
            selected_samples = self._split_train_validation(
                samples=base_samples,
                split=split,
                validation_ratio=validation_ratio,
                seed=seed
            )
        self.samples = selected_samples

    def _collect_samples(self, root: Path, split="train"):
        """collect all samples (data points) from a given split"""
        split_dir = root / split
        samples = []
        for file_path in sorted(split_dir.glob("*/*.*")): #e.g. train/img0.PIL
            label_name = file_path.parent.name.upper()
            if label_name not in {"FAKE", "REAL"}:
                raise ValueError("Unexpected label name: " + label_name)
            label = 1 if label_name == "FAKE" else 0
            samples.append((file_path, label))
        if not samples:
            raise ValueError(f"No image files found under: {split_dir}")
        return samples

    def _split_train_validation(self, samples, split, validation_ratio, seed):
        """partition train into train and validation"""
        num_samples = len(samples)
        indices = torch.randperm(
            num_samples,
            generator=torch.Generator().manual_seed(seed)
        ).tolist()
        split_idx = int(num_samples * (1 - validation_ratio))
        if split == "train":
            selected_indices = indices[:split_idx]
        else:  # validation
            selected_indices = indices[split_idx:]
        return [samples[i] for i in selected_indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image, label

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
