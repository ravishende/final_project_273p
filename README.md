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
