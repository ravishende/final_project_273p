import io
import random
import kagglehub
from pathlib import Path

import kagglehub
import torch
import torchvision.transforms.functional as TF

from PIL import Image
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Transforms
class RandomJPEGCompression:
    def __init__(self, quality=50, p=0.2):
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
    def __init__(self, std=0.3, p=0.2):
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        x = x + noise
        return x.clamp(0.0, 1.0)


class FixedBrightness:
    def __init__(self, factor=0.5, p=0.2):
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

# Dataset wrappers
RAJARSHI_SOURCE_MAP = {
    0: "real",
    1: "src_1",
    2: "src_2",
    3: "src_3",
    4: "src_4",
    5: "src_5",
}


class RajarshiDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        self.labels = [int(x) for x in dataset["Label_A"]]
        self.dataset_names = ["rajarshi"] * len(dataset)
        self.source_names = [RAJARSHI_SOURCE_MAP[int(x)] for x in dataset["Label_B"]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["Image"].convert("RGB")
        label = int(row["Label_A"])
        source_name = RAJARSHI_SOURCE_MAP[int(row["Label_B"])]

        if self.transform:
            image = self.transform(image)

        return image, label, "rajarshi", source_name

class HemgDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

        # original label assumed: 1 = real, 0 = fake
        # convert to: 0 = real, 1 = fake
        self.labels = [int(1 - int(x)) for x in dataset["label"]]
        self.dataset_names = ["hemg"] * len(dataset)
        self.source_names = ["unknown"] * len(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = row["image"].convert("RGB")
        label = int(1 - int(row["label"]))

        if self.transform:
            image = self.transform(image)

        return image, label, "hemg", "unknown"

class CIFAKEDataset(Dataset):
    def __init__(self, split="train", transform=None):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.transform = transform

        root = Path(
            kagglehub.dataset_download(
                "birdy654/cifake-real-and-ai-generated-synthetic-images"
            )
        )

        self.samples = self._collect_samples(root, split)
        self.labels = [label for _, label in self.samples]
        self.dataset_names = ["cifake"] * len(self.samples)
        self.source_names = ["unknown"] * len(self.samples)

    def _collect_samples(self, root: Path, split="train"):
        """collect all samples (data points) from a given split"""
        split_dir = root / split
        samples = []

        for file_path in sorted(split_dir.glob("*/*.*")):
            label_name = file_path.parent.name.upper()
            if label_name not in {"FAKE", "REAL"}:
                raise ValueError(f"Unexpected label name: {label_name}")

            label = 1 if label_name == "FAKE" else 0
            samples.append((file_path, label))

        if not samples:
            raise ValueError(f"No image files found under: {split_dir}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)

        return image, label, "cifake", "unknown"

# Split helper functions
def stratified_split_indices(labels, train_ratio, val_ratio, seed=0):
    indices = list(range(len(labels)))
    test_ratio = 1.0 - train_ratio - val_ratio

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = [labels[i] for i in train_val_idx]
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_relative_ratio,
        random_state=seed,
        stratify=train_val_labels,
    )

    return train_idx, val_idx, test_idx


class SubsetWithMetadata(Dataset):
    """wrapper around a dataset with selected indices"""
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = indices

        self.labels = [base_dataset.labels[i] for i in indices]
        self.dataset_names = [base_dataset.dataset_names[i] for i in indices]
        self.source_names = [base_dataset.source_names[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]

# Aggregate sampler
def build_aggregate_sampler(dataset, dataset_weight_map=None):
    """
    dataset must expose:
      - labels
      - dataset_names
    """
    if dataset_weight_map is None:
        dataset_weight_map = {
            "rajarshi": 1.0,
            "cifake": 1.0,
            "hemg": 1.0,
        }

    labels = dataset.labels
    dataset_names = dataset.dataset_names

    class_counts = {}
    for y in labels:
        class_counts[y] = class_counts.get(y, 0) + 1

    class_weights = {
        cls: 1.0 / count for cls, count in class_counts.items()
    }

    sample_weights = []
    for y, dname in zip(labels, dataset_names):
        w = class_weights[y] * dataset_weight_map[dname]
        sample_weights.append(w)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler

# Aggregate dataset wrapper
class AggregateDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.concat = ConcatDataset(datasets)

        self.labels = []
        self.dataset_names = []
        self.source_names = []

        for ds in datasets:
            self.labels.extend(ds.labels)
            self.dataset_names.extend(ds.dataset_names)
            self.source_names.extend(ds.source_names)

    def __len__(self):
        return len(self.concat)

    def __getitem__(self, idx):
        return self.concat[idx]

# Dataset builders
def build_individual_datasets(cfg):
    train_transform, eval_transform = build_transforms(cfg.image_size)

    # Rajarshi
    raj_train_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="train",
    )
    raj_val_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="validation",
    )
    raj_test_hf = load_dataset(
        "Rajarshi-Roy-research/Defactify_Image_Dataset",
        split="test",
    )

    raj_train = RajarshiDataset(raj_train_hf, transform=train_transform)
    raj_val = RajarshiDataset(raj_val_hf, transform=eval_transform)
    raj_test = RajarshiDataset(raj_test_hf, transform=eval_transform)

    # CIFAKE
    cifake_full_train = CIFAKEDataset(split="train", transform=None)
    cifake_test = CIFAKEDataset(split="test", transform=eval_transform)

    indices = list(range(len(cifake_full_train.labels)))

    cifake_train_idx, cifake_val_idx  = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=cifake_full_train.labels,
    )

    cifake_train_base = CIFAKEDataset(split="train", transform=train_transform)
    cifake_val_base = CIFAKEDataset(split="train", transform=eval_transform)

    cifake_train = SubsetWithMetadata(cifake_train_base, cifake_train_idx)
    cifake_val = SubsetWithMetadata(cifake_val_base, cifake_val_idx)


    # Hemg
    hemg_hf = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")  # replace with actual HF dataset name
    hemg_full_base_train = HemgDataset(hemg_hf, transform=train_transform)
    hemg_full_base_eval = HemgDataset(hemg_hf, transform=eval_transform)

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels=hemg_full_base_train.labels,
        train_ratio=0.8,
        val_ratio=0.1,
        seed=cfg.seed,
    )

    hemg_train = SubsetWithMetadata(hemg_full_base_train, train_idx)
    hemg_val = SubsetWithMetadata(hemg_full_base_eval, val_idx)
    hemg_test = SubsetWithMetadata(hemg_full_base_eval, test_idx)

    return {
        "rajarshi": {"train": raj_train, "val": raj_val, "test": raj_test},
        "cifake": {"train": cifake_train, "val": cifake_val, "test": cifake_test},
        "hemg": {"train": hemg_train, "val": hemg_val, "test": hemg_test},
    }

def build_aggregate_dataloaders(cfg):
    datasets = build_individual_datasets(cfg)

    dataset_weight_map = {
        "rajarshi": 1.0,
        "cifake": 1.0,
        "hemg": 1.0,
    }


    train_dataset = AggregateDataset([
        datasets["rajarshi"]["train"],
        datasets["cifake"]["train"],
        datasets["hemg"]["train"],
    ])

    val_dataset = AggregateDataset([
        datasets["rajarshi"]["val"],
        datasets["cifake"]["val"],
        datasets["hemg"]["val"],
    ])

    persistent = cfg.persistent_workers if cfg.num_workers > 0 else False

    train_sampler = build_aggregate_sampler(
        train_dataset,
        dataset_weight_map=dataset_weight_map,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        shuffle=False,
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

    test_loaders = {
        name: DataLoader(
            datasets[name]["test"],
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=persistent,
        )
        for name in datasets
    }

    return train_loader, val_loader, test_loaders