import os
import random
import numpy as np
import wandb
import torch
import torch.nn as nn

from dataclasses import asdict
from torch.optim import AdamW

from config import Config
from final_project_273p.src.data import build_aggregate_dataloaders
from final_project_273p.src.model import ResNet18RGB, ResNet18FFT1C, ResNet18RealArtifactNet
from eval import (
    evaluate,
    evaluate_per_dataset,
    evaluate_per_dataset_source,
    print_grouped_results,
)

def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model_name == "rgb":
        return ResNet18RGB(
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
        )

    if cfg.model_name == "fft":
        return ResNet18FFT1C(
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
        )

    if cfg.model_name == "real_artifact_net":
        return ResNet18RealArtifactNet(
            pretrained=cfg.pretrained,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
        )

    raise ValueError(f"Unknown model_name: {cfg.model_name}")


def build_optimizer(model, cfg):
    """
    we will use differential learning rate: 
    https://blog.slavv.com/differential-learning-rates-59eff5209a4f

    where the backbone (ResNet18) 's learning rate is slower than the head (what we added)
    the idea is that we don't want to change the earlier layers (pretrained) too much
    """
    if cfg.model_name in ["rgb", "fft"]:
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "backbone.fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        if len(head_params) == 0:
            raise ValueError(f"No head params found for model_name={cfg.model_name}")

        return AdamW([
            {"params": backbone_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)

    if cfg.model_name == "real_artifact_net":
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if (
                "artifact_score" in name
                or "realness_score" in name
                or "artifact_norm" in name
                or "realness_norm" in name
                or "log_alpha" in name
                or "log_beta" in name
                or "bias" in name
            ):
                head_params.append(param)
            else:
                backbone_params.append(param)

        if len(head_params) == 0:
            raise ValueError(f"No head params found for model_name={cfg.model_name}")

        return AdamW([
            {"params": backbone_params, "lr": cfg.lr * 0.1},
            {"params": head_params, "lr": cfg.lr},
        ], weight_decay=cfg.weight_decay)

    raise ValueError(f"Unknown model_name: {cfg.model_name}")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

    return {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
    }


def is_better(current, best, mode="max", min_delta=1e-4):
    if mode == "max":
        return current > best + min_delta
    if mode == "min":
        return current < best - min_delta
    raise ValueError(f"Unknown mode: {mode}")


def main():
    cfg = Config(
        model_name="real_artifact_net",
        run_name="resnet18-real_artifact_net-aggregate-final"
    )

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, test_loaders = build_aggregate_dataloaders(cfg)

    run = wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=asdict(cfg),
    )

    model = build_model(cfg).to(device)

    # 7000 "real"s and 31800 "fake"s in training set result in class imbalance
    # so we use a weighted loss function
    # can change this depends on training set distribution
    # class_counts = torch.tensor([7000, 31800], dtype=torch.float32, device=device) 
    # class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    # criterion = nn.CrossEntropyLoss(
    #     weight=class_weights
    # )

    criterion = nn.CrossEntropyLoss()

    optimizer = build_optimizer(model, cfg)

    best_ckpt_path = os.path.join(cfg.save_dir, f"{cfg.run_name}.pt")
    best_epoch = -1
    best_metric = float("-inf") if cfg.monitor_mode == "max" else float("inf")
    epochs_without_improve = 0

    for epoch in range(cfg.max_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=cfg.num_classes,
        )

        current_metric = val_metrics[cfg.monitor_metric]

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/acc": train_metrics["acc"],
            "val/loss": val_metrics["loss"],
            "val/acc": val_metrics["acc"],
            "val/precision": val_metrics["precision"],
            "val/recall": val_metrics["recall"],
            "val/f1": val_metrics["f1"],
            "val/auc": val_metrics["auc"],
        })

        print(
            f"Epoch {epoch+1:02d} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if is_better(
            current=current_metric,
            best=best_metric,
            mode=cfg.monitor_mode,
            min_delta=cfg.min_delta,
        ):
            best_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improve = 0

            save_checkpoint(model, best_ckpt_path)
            run.summary[f"best_{cfg.monitor_metric}"] = best_metric
            run.summary["best_epoch"] = best_epoch

        else:
            epochs_without_improve += 1

        if (epoch + 1) >= cfg.min_epochs and epochs_without_improve >= cfg.patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {best_ckpt_path}")

    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))

    best_val_metrics = evaluate(
        model,
        val_loader,
        criterion,
        device,
        num_classes=cfg.num_classes,
    )

    print(
        f"\nReloaded best checkpoint VAL | "
        f"acc={best_val_metrics['acc']:.4f} "
        f"f1={best_val_metrics['f1']:.4f} "
        f"auc={best_val_metrics['auc']:.4f}"
    )

    # grouped validation only on best checkpoint
    val_per_dataset = evaluate_per_dataset(
        model,
        val_loader,
        criterion,
        device,
        num_classes=cfg.num_classes,
    )
    val_per_dataset_source = evaluate_per_dataset_source(
        model,
        val_loader,
        criterion,
        device,
    )
    print_grouped_results("Validation per-dataset [best checkpoint]", val_per_dataset)
    print_grouped_results("Validation per-dataset-source [best checkpoint]", val_per_dataset_source)
    for group_name, metrics in val_per_dataset.items():
        wandb.log({
            f"best_val_per_dataset/{group_name}_acc": metrics["acc"],
            f"best_val_per_dataset/{group_name}_f1": metrics["f1"],
            f"best_val_per_dataset/{group_name}_auc": metrics["auc"],
            f"best_val_per_dataset/{group_name}_loss": metrics["loss"],
            "epoch": best_epoch,
        })
    for group_name, metrics in val_per_dataset_source.items():
        wandb.log({
            f"best_val_per_dataset_source/{group_name}_acc": metrics["acc"],
            f"best_val_per_dataset_source/{group_name}_err": metrics["error_rate"],
            f"best_val_per_dataset_source/{group_name}_loss": metrics["loss"],
            "epoch": best_epoch,
        })

    all_test_results = {}

    for dataset_name, test_loader in test_loaders.items():
        test_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            num_classes=cfg.num_classes,
        )
        if dataset_name == "rajarshi":
            test_per_dataset_source = evaluate_per_dataset_source(
                model,
                test_loader,
                criterion,
                device,
            )
            print_grouped_results(f"Test per-dataset-source [{dataset_name}]", test_per_dataset_source)

        all_test_results[dataset_name] = test_metrics

        print(
            f"\n[{dataset_name}] "
            f"acc={test_metrics['acc']:.4f} "
            f"precision={test_metrics['precision']:.4f} "
            f"recall={test_metrics['recall']:.4f} "
            f"f1={test_metrics['f1']:.4f} "
            f"auc={test_metrics['auc']:.4f}"
        )

        wandb.log({
            f"test/{dataset_name}/loss": test_metrics["loss"],
            f"test/{dataset_name}/acc": test_metrics["acc"],
            f"test/{dataset_name}/precision": test_metrics["precision"],
            f"test/{dataset_name}/recall": test_metrics["recall"],
            f"test/{dataset_name}/f1": test_metrics["f1"],
            f"test/{dataset_name}/auc": test_metrics["auc"],
            f"test/{dataset_name}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=test_metrics["labels"].numpy(),
                preds=test_metrics["preds"].numpy(),
                class_names=["real", "fake"],
            ),
        })

        run.summary[f"test_{dataset_name}_acc"] = test_metrics["acc"]
        run.summary[f"test_{dataset_name}_precision"] = test_metrics["precision"]
        run.summary[f"test_{dataset_name}_recall"] = test_metrics["recall"]
        run.summary[f"test_{dataset_name}_f1"] = test_metrics["f1"]
        run.summary[f"test_{dataset_name}_auc"] = test_metrics["auc"]

    artifact = wandb.Artifact(name=cfg.run_name, type="model")
    artifact.add_file(best_ckpt_path)
    wandb.log_artifact(artifact)

    wandb.finish()

    print(f"\nBest val {cfg.monitor_metric}: {best_metric:.4f} at epoch {best_epoch}")
    for dataset_name, metrics in all_test_results.items():
        print(
            f"[FINAL TEST] {dataset_name}: "
            f"acc={metrics['acc']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"auc={metrics['auc']:.4f}"
        )
    

if __name__ == "__main__":
    main()