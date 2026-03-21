# COMPSCI 273P Final Project

# Project Overview

# Setup

Set up a virtual environment

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

Required dependencies:

- python version >= 3.9
- required libraries can be viewed in `requirements.txt`

# Modeling

## How to train the model

Go into the `src` directory

1. run `train.py`
   - This will require a GPU

## How to evaluate the model

After running train.py, make an account with Weights & Biases. Metrics and graphs will be written there to evaluate, such as precision, recall, loss, accuracy, f1, AUC, etc...

## Expected outputs

Example Output from the following config:

```py
cfg = Config(
        model_name="real_artifact_net",
        run_name="resnet18-real_artifact_net-aggregate-final"
    )
```

```
Epoch 01 | train_loss=0.2616 train_acc=0.8820 | val_loss=0.1960 val_acc=0.9142 val_f1=0.9161 val_auc=0.9752
Epoch 02 | train_loss=0.1575 train_acc=0.9339 | val_loss=0.1443 val_acc=0.9415 val_f1=0.9413 val_auc=0.9868
Epoch 03 | train_loss=0.1224 train_acc=0.9496 | val_loss=0.1263 val_acc=0.9477 val_f1=0.9491 val_auc=0.9899
Epoch 04 | train_loss=0.1019 train_acc=0.9585 | val_loss=0.1239 val_acc=0.9506 val_f1=0.9521 val_auc=0.9907
Epoch 05 | train_loss=0.0846 train_acc=0.9657 | val_loss=0.1371 val_acc=0.9447 val_f1=0.9484 val_auc=0.9912
Epoch 06 | train_loss=0.0738 train_acc=0.9702 | val_loss=0.1180 val_acc=0.9563 val_f1=0.9578 val_auc=0.9928
Epoch 07 | train_loss=0.0662 train_acc=0.9730 | val_loss=0.1238 val_acc=0.9555 val_f1=0.9566 val_auc=0.9920
Epoch 08 | train_loss=0.0591 train_acc=0.9759 | val_loss=0.1178 val_acc=0.9602 val_f1=0.9607 val_auc=0.9933
Epoch 09 | train_loss=0.0544 train_acc=0.9776 | val_loss=0.1184 val_acc=0.9627 val_f1=0.9623 val_auc=0.9936
Epoch 10 | train_loss=0.0495 train_acc=0.9799 | val_loss=0.1235 val_acc=0.9621 val_f1=0.9631 val_auc=0.9935
Epoch 11 | train_loss=0.0473 train_acc=0.9806 | val_loss=0.1304 val_acc=0.9618 val_f1=0.9618 val_auc=0.9936
Epoch 12 | train_loss=0.0450 train_acc=0.9815 | val_loss=0.1277 val_acc=0.9622 val_f1=0.9626 val_auc=0.9942
Epoch 13 | train_loss=0.0416 train_acc=0.9830 | val_loss=0.1309 val_acc=0.9636 val_f1=0.9635 val_auc=0.9940
Epoch 14 | train_loss=0.0407 train_acc=0.9837 | val_loss=0.1364 val_acc=0.9616 val_f1=0.9637 val_auc=0.9937
Epoch 15 | train_loss=0.0384 train_acc=0.9845 | val_loss=0.1459 val_acc=0.9587 val_f1=0.9604 val_auc=0.9938
Epoch 16 | train_loss=0.0368 train_acc=0.9851 | val_loss=0.1396 val_acc=0.9633 val_f1=0.9649 val_auc=0.9941
Epoch 17 | train_loss=0.0351 train_acc=0.9857 | val_loss=0.1357 val_acc=0.9642 val_f1=0.9652 val_auc=0.9946
Epoch 18 | train_loss=0.0340 train_acc=0.9864 | val_loss=0.1454 val_acc=0.9640 val_f1=0.9646 val_auc=0.9943
Epoch 19 | train_loss=0.0335 train_acc=0.9866 | val_loss=0.1504 val_acc=0.9626 val_f1=0.9648 val_auc=0.9938
Epoch 20 | train_loss=0.0315 train_acc=0.9871 | val_loss=0.1334 val_acc=0.9665 val_f1=0.9667 val_auc=0.9950
Epoch 21 | train_loss=0.0301 train_acc=0.9880 | val_loss=0.1439 val_acc=0.9665 val_f1=0.9681 val_auc=0.9953
Epoch 22 | train_loss=0.0292 train_acc=0.9882 | val_loss=0.1392 val_acc=0.9663 val_f1=0.9679 val_auc=0.9951
Epoch 23 | train_loss=0.0285 train_acc=0.9884 | val_loss=0.1418 val_acc=0.9673 val_f1=0.9666 val_auc=0.9945
Epoch 24 | train_loss=0.0273 train_acc=0.9890 | val_loss=0.1394 val_acc=0.9683 val_f1=0.9693 val_auc=0.9952
Epoch 25 | train_loss=0.0271 train_acc=0.9892 | val_loss=0.1448 val_acc=0.9659 val_f1=0.9669 val_auc=0.9952
Epoch 26 | train_loss=0.0267 train_acc=0.9891 | val_loss=0.1358 val_acc=0.9678 val_f1=0.9685 val_auc=0.9953
Early stopping triggered at epoch 26

Reloaded best checkpoint VAL | acc=0.9665 f1=0.9681 auc=0.9953

=== Validation per-dataset [best checkpoint] ===
group                           count      acc       f1      auc       loss
rajarshi                         9000   0.8479   0.8669   0.9612     0.3344
cifake                          20000   0.9872   0.9872   0.9994     0.0544
hemg                            15272   0.9677   0.9679   0.9951     0.1487

=== Validation per-dataset-source [best checkpoint] ===
group                           count      acc      err       loss
rajarshi::src_3                  1500   0.4880   0.5120     0.3307
rajarshi::real                   1500   0.3623   0.6377     0.3398
rajarshi::src_1                  1500   0.4857   0.5143     0.3354
rajarshi::src_4                  1500   0.4957   0.5043     0.3364
rajarshi::src_2                  1500   0.4893   0.5107     0.3313
rajarshi::src_5                  1500   0.4690   0.5310     0.3326
cifake::unknown                 20000   0.9872   0.0128     0.0544
hemg::unknown                   15272   0.9677   0.0323     0.1487

=== Test per-dataset-source [rajarshi] ===
group                           count      acc      err       loss
rajarshi::src_5                  7500   0.4345   0.5655     1.6507
rajarshi::real                   7500   0.2062   0.7938     1.7041
rajarshi::src_1                  7500   0.4416   0.5584     1.6703
rajarshi::src_2                  7500   0.4425   0.5575     1.6615
rajarshi::src_3                  7500   0.4407   0.5593     1.6632
rajarshi::src_4                  7500   0.4496   0.5504     1.6455

[rajarshi] acc=0.6480 precision=0.6486 recall=0.6480 f1=0.6483 auc=0.6824

[cifake] acc=0.9899 precision=0.9900 recall=0.9899 f1=0.9898 auc=0.9993

[hemg] acc=0.9652 precision=0.9657 recall=0.9652 f1=0.9655 auc=0.9947
```

# Datasets

## Where to download the datasets

Rajarshi: Hugging Face Real vs LLM (variety) Dataset:
`https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset`

Hemg: Hugging Face Real vs AI Dataset:
`https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets`

CIFAKE: Kaggle AI vs Real Images:
`https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/data`

## How to preprocess the data

We created DataLoaders to iterate over the data. We updated the labels to ensure that 1 means AI-generated and 0 means real across all 3 data sets.

## How to reproduce our results

Go into the `src` directory

1. run `train.py`
   - This will require a GPU
2. See resulting graphs in the wandb report
3. Get the results csv from wandb. Put it in the `src` directory. Rename it to `stats.csv`. Go into the `src` directory. Run all cells in `tabulate_results.ipynb`
