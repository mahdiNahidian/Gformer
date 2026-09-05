# Gformer: Gradual ProbSparse Attention for Long-Sequence Time-Series Forecasting

## Overview

This repository provides the implementation of Gformer, a
Transformer-based framework for long-sequence time-series forecasting
with Gradual ProbSparse Attention (G-PSA).

Gformer introduces a gradual attention reduction strategy that
progressively identifies and removes less informative queries during
training while preserving important temporal dependencies. The framework
is designed to improve the efficiency of long-sequence forecasting while
maintaining predictive performance.

The implementation includes a Transformer-based encoder-decoder
architecture, temporal embeddings, attention mechanisms, and the
proposed G-PSA module.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The code is developed using PyTorch-based environments.

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

## Dataset Preparation

The experiments are conducted on six datasets covering energy,
meteorological, and financial time-series forecasting scenarios.

The datasets used in the experiments are:

- ETTh1
- ETTh2
- Weather
- Electricity (ECL)
- AAPL
- AMZN

Place the datasets in the data directory according to the format
expected by the provided data loader.

The dataset files should follow the required CSV format.

## Training and Evaluation

Example training command:

```bash
python main_gformer.py \
--model gformer \
--data ETTh1 \
--features M \
--seq_len 96 \
--label_len 48 \
--pred_len 24 \
--attn custom
```

Important arguments:

- `--model`: model architecture (`gformer` or `gformerstack`)
- `--data`: dataset name
- `--features`: forecasting setting (`M`, `S`, or `MS`)
- `--seq_len`: input sequence length
- `--label_len`: decoder start token length
- `--pred_len`: forecasting horizon
- `--attn`: attention mechanism (`custom` uses G-PSA)

## Example

An example notebook is provided in the `examples/` directory to
demonstrate the usage of Gformer on the AAPL forecasting scenario.

## Citation

The citation information will be added after the paper publication.
