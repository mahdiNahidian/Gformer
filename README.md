# Gformer: Gradual ProbSparse Attention for Long-Sequence Time-Series Forecasting

## Overview

This repository provides the implementation of Gformer, a framework for
long-sequence time-series forecasting with Gradual ProbSparse Attention
(G-PSA).

Gformer introduces a gradual attention reduction strategy that
progressively identifies and removes less informative queries during
training while preserving important temporal dependencies. The approach
is designed to improve the efficiency of long-sequence forecasting while
maintaining predictive performance.

The implementation is built on a Transformer-based time-series
forecasting framework and retains standard components such as temporal
embeddings, encoder-decoder architecture, and attention masking.

------------------------------------------------------------------------

## Installation

Clone the repository and install the required dependencies:

``` bash
pip install -r requirements.txt
```

The code has been developed with PyTorch-based environments.

------------------------------------------------------------------------

## Repository Structure

    Gformer/
    │
    ├── main_gformer.py
    ├── requirements.txt
    │
    ├── models/
    │   ├── gformer.py
    │   ├── attention.py
    │   ├── encoder.py
    │   ├── decoder.py
    │   └── embed.py
    │
    ├── exp/
    │   ├── exp_basic.py
    │   └── exp_gformer.py
    │
    ├── data/
    │   └── data_loader.py
    │
    ├── utils/
    │   ├── tools.py
    │   ├── metrics.py
    │   ├── masking.py
    │   └── timefeatures.py
    │
    └── examples/
        └── Gformer_AAPL.ipynb

------------------------------------------------------------------------

## Dataset Preparation

The data loader supports common long-sequence time-series forecasting
datasets.

Place the datasets in the data directory according to the expected
format.

Supported datasets include:

-   ETTh1
-   ETTh2
-   ETTm1
-   ETTm2
-   Electricity (ECL)
-   Weather (WTH)
-   Solar

The dataset files should follow the required CSV format used by the
provided data loader.

------------------------------------------------------------------------

## Training and Evaluation

Example training command:

``` bash
python main_gformer.py \
--model gformer \
--data ETTh1 \
--features M \
--seq_len 96 \
--label_len 48 \
--pred_len 24 \
--attn custom
```

The script performs training and evaluation using the specified
forecasting configuration.

Important arguments:

-   `--model`: model architecture (`gformer` or `gformerstack`)
-   `--data`: dataset name
-   `--features`: forecasting setting (`M`, `S`, or `MS`)
-   `--seq_len`: input sequence length
-   `--label_len`: decoder start token length
-   `--pred_len`: forecasting horizon
-   `--attn`: attention mechanism (`custom` uses G-PSA)

------------------------------------------------------------------------

## Example

An example notebook is provided in the `examples/` directory to
demonstrate the usage of Gformer on an additional forecasting scenario.

------------------------------------------------------------------------

## Citation

The citation information will be added after the paper publication.
