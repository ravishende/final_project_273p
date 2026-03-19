import torch
from collections import defaultdict
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
    BinaryAUROC,
)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2):
    model.eval()

    total_loss = 0.0
    total_count = 0

    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    precision_metric = MulticlassPrecision(
        num_classes=num_classes,
        average="macro",
    ).to(device)
    recall_metric = MulticlassRecall(
        num_classes=num_classes,
        average="macro",
    ).to(device)
    f1_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro",
    ).to(device)
    cm_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    auroc_metric = BinaryAUROC().to(device)

    all_probs = []
    all_preds = []
    all_labels = []

    for batch in loader:
        if len(batch) == 4:
            images, labels, _, _ = batch
        elif len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        fake_probs = probs[:, 1]
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

        acc_metric.update(preds, labels)
        precision_metric.update(preds, labels)
        recall_metric.update(preds, labels)
        f1_metric.update(preds, labels)
        cm_metric.update(preds, labels)
        auroc_metric.update(fake_probs, labels)

        all_probs.append(probs.detach().cpu())
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = acc_metric.compute().item()
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1 = f1_metric.compute().item()

    try:
        auc = auroc_metric.compute().item()
    except ValueError:
        auc = float("nan")

    cm = cm_metric.compute().detach().cpu().numpy()

    return {
        "loss": total_loss / total_count,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
    }

# @torch.no_grad()
# def evaluate_per_source(model, loader, criterion, device, num_classes=2, source_names=None):
#     """
#     This is for RajarshiDataset only, we evaluate the model and output the metrics 
#     from each source (real, Stable Diffusion ...)
#     """
#     model.eval()

#     per_source_preds = defaultdict(list)
#     per_source_labels = defaultdict(list)
#     per_source_losses = defaultdict(list)

#     for images, labels_a, labels_b in loader:
#         images = images.to(device, non_blocking=True)
#         labels_a = labels_a.to(device, non_blocking=True)
#         labels_b = labels_b.to(device, non_blocking=True)

#         logits = model(images)
#         loss = criterion(logits, labels_a)

#         preds = logits.argmax(dim=1)

#         # move to cpu for grouping
#         preds = preds.detach().cpu()
#         labels_a = labels_a.detach().cpu()
#         labels_b = labels_b.detach().cpu()

#         batch_size = labels_a.size(0)
#         batch_loss = loss.item()

#         for i in range(batch_size):
#             src = int(labels_b[i].item())
#             per_source_preds[src].append(int(preds[i].item()))
#             per_source_labels[src].append(int(labels_a[i].item()))
#             per_source_losses[src].append(batch_loss)

#     results = {}

#     for src in sorted(per_source_labels.keys()):
#         y_true = torch.tensor(per_source_labels[src], dtype=torch.long)
#         y_pred = torch.tensor(per_source_preds[src], dtype=torch.long)

#         acc_metric = MulticlassAccuracy(num_classes=num_classes)
#         acc = acc_metric(y_pred, y_true).item()

#         avg_loss = sum(per_source_losses[src]) / len(per_source_losses[src])

#         src_name = source_names[src] if source_names is not None else str(src)

#         results[src_name] = {
#             "count": len(y_true),
#             "loss": avg_loss,
#             "acc": acc,
#             "error_rate": 1.0 - acc,
#         }

#     return results


# def print_per_source_results(title, results):
#     print(f"\n=== {title} ===")
#     print(f"{'source':<15} {'count':>8} {'acc':>8} {'err':>8} {'loss':>10}")
#     for source, metrics in results.items():
#         print(
#             f"{source:<15} "
#             f"{metrics['count']:>8d} "
#             f"{metrics['acc']:>8.4f} "
#             f"{metrics['error_rate']:>8.4f} "
#             f"{metrics['loss']:>10.4f}"
#         )


@torch.no_grad()
def evaluate_per_dataset(model, loader, criterion, device, num_classes=2):
    """
    loader batches:
      images, labels, dataset_names, source_names
    """
    model.eval()

    groups = defaultdict(lambda: {
        "preds": [],
        "labels": [],
        "fake_probs": [],
        "losses": [],
    })

    for images, labels, dataset_names, source_names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        fake_probs = probs[:, 1]
        preds = logits.argmax(dim=1)

        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()
        fake_probs_cpu = fake_probs.detach().cpu()

        for i in range(labels.size(0)):
            dname = dataset_names[i]
            groups[dname]["preds"].append(int(preds_cpu[i].item()))
            groups[dname]["labels"].append(int(labels_cpu[i].item()))
            groups[dname]["fake_probs"].append(float(fake_probs_cpu[i].item()))
            groups[dname]["losses"].append(loss.item())

    results = {}

    for dname, g in groups.items():
        preds = torch.tensor(g["preds"], dtype=torch.long)
        labels = torch.tensor(g["labels"], dtype=torch.long)
        fake_probs = torch.tensor(g["fake_probs"], dtype=torch.float32)

        acc = MulticlassAccuracy(num_classes=num_classes)(preds, labels).item()
        precision = MulticlassPrecision(num_classes=num_classes, average="macro")(preds, labels).item()
        recall = MulticlassRecall(num_classes=num_classes, average="macro")(preds, labels).item()
        f1 = MulticlassF1Score(num_classes=num_classes, average="macro")(preds, labels).item()

        try:
            auc = BinaryAUROC()(fake_probs, labels).item()
        except ValueError:
            auc = float("nan")

        avg_loss = sum(g["losses"]) / len(g["losses"])

        results[dname] = {
            "count": len(labels),
            "loss": avg_loss,
            "acc": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    return results

@torch.no_grad()
def evaluate_per_dataset_source(model, loader, criterion, device):
    """
    For aggregate dataset:
      group by (dataset_name, source_name)
    """
    model.eval()

    groups = defaultdict(lambda: {
        "preds": [],
        "labels": [],
        "losses": [],
    })

    for images, labels, dataset_names, source_names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        preds_cpu = preds.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for i in range(labels.size(0)):
            key = f"{dataset_names[i]}::{source_names[i]}"
            groups[key]["preds"].append(int(preds_cpu[i].item()))
            groups[key]["labels"].append(int(labels_cpu[i].item()))
            groups[key]["losses"].append(loss.item())

    results = {}

    for key, g in groups.items():
        preds = torch.tensor(g["preds"], dtype=torch.long)
        labels = torch.tensor(g["labels"], dtype=torch.long)

        acc = MulticlassAccuracy(num_classes=2)(preds, labels).item()
        avg_loss = sum(g["losses"]) / len(g["losses"])

        results[key] = {
            "count": len(labels),
            "loss": avg_loss,
            "acc": acc,
            "error_rate": 1.0 - acc,
        }

    return results


def print_grouped_results(title, results):
    print(f"\n=== {title} ===")
    first = next(iter(results.values()))
    if "auc" in first:
        print(f"{'group':<28} {'count':>8} {'acc':>8} {'f1':>8} {'auc':>8} {'loss':>10}")
        for k, v in results.items():
            print(
                f"{k:<28} "
                f"{v['count']:>8d} "
                f"{v['acc']:>8.4f} "
                f"{v['f1']:>8.4f} "
                f"{v['auc']:>8.4f} "
                f"{v['loss']:>10.4f}"
            )
    else:
        print(f"{'group':<28} {'count':>8} {'acc':>8} {'err':>8} {'loss':>10}")
        for k, v in results.items():
            print(
                f"{k:<28} "
                f"{v['count']:>8d} "
                f"{v['acc']:>8.4f} "
                f"{v['error_rate']:>8.4f} "
                f"{v['loss']:>10.4f}"
            )