# pyFAST: Flexible, Advanced Framework for Multi-source and Sparse Time Series Analysis in PyTorch

[![Software Overview Figure](overview.svg)](overview.svg) 


pyFAST (Forecasting And time-Series in PyTorch) is a **research-driven, modular Python framework** built for **advanced and efficient time series analysis**, especially excelling in **multi-source and sparse data scenarios**.  Leveraging PyTorch, pyFAST provides a unified and flexible platform for forecasting, imputation, and generative modeling, integrating cutting-edge **LLM-inspired architectures**, Variational Autoencoders, and classical time series models.

**Unlock the Power of pyFAST for:**

*   **Alignment-Free Multi-source Time Series Analysis:**  Process and fuse data from diverse sources without the need for strict temporal alignment, inspired by Large Language Model principles.
*   **Native Sparse Time Series Forecasting:**  Effectively handle and forecast sparse time series data with specialized metrics and loss functions, addressing a critical gap in existing libraries.
*   **Rapid Research Prototyping:**  Experiment and prototype novel time series models and techniques with unparalleled flexibility and modularity.
*   **Seamless Customization and Extensibility:**  Tailor and extend the library to your specific research or application needs with its component-based modular design.
*   **High Performance and Scalability:**  Benefit from optimized PyTorch implementations and multi-device acceleration for efficient handling of large datasets and complex models.

**Key Capabilities:**

*   **Pioneering LLM-Inspired Models:** First-of-its-kind adaptations of Large Language Models specifically for alignment-free multi-source time series forecasting.
*   **Native Sparse Data Support:** Comprehensive support for sparse time series, including specialized metrics, loss functions, and efficient data handling.
*   **Flexible Multi-source Data Fusion:**  Integrate and analyze time series data from diverse, potentially misaligned sources.
*   **Extensive Model Library:**  Includes a broad range of classical, deep learning (Transformers, RNNs, CNNs, GNNs), and generative time series models for both multivariate (MTS) and univariate (UTS) data.
*   **Modular and Extensible Architecture:**  Component-based design enables easy customization, extension, and combination of modules.
*   **Streamlined Training Pipeline:** `Trainer` class simplifies model training with built-in validation, early stopping, checkpointing, and multi-device support.
*   **Comprehensive Evaluation Suite:** Includes a wide array of standard and sparse-specific evaluation metrics via the `Evaluator` class.
*   **Built-in Generative Modeling:** Dedicated module for time series Variational Autoencoders (VAEs), including Transformer-based VAEs.
*   **Reproducibility Focus:** Utilities like `initial_seed()` ensure experiment reproducibility.

**Explore the Core Modules (See Figure Above):**

As depicted in the Software Overview Diagram above (Figure 1), pyFAST's `fast/` library is structured into five core modules, ensuring a cohesive and versatile framework:

*   **`data/` Module:**  Handles data loading, preprocessing, and dataset creation for SST, SMT, MTM, and BDP data scenarios.  Key features include efficient sparse data handling, multi-source data integration, scaling methods, patching, and data splitting utilities.
*   **`model/` Module:**  Houses a diverse collection of time series models, categorized into `uts/` (univariate), `mts/` (multivariate), and `base/` (building blocks) submodules. Includes classical models, deep learning architectures (CNNs, RNNs, Transformers, GNNs), fusion models, and generative models.
*   **`train.py` Module:**  Provides the `Trainer` class to streamline the entire model training pipeline. Features include device management, model compilation, optimizer and scheduler management, training loop, validation, early stopping, checkpointing, and visualization integration.
*   **`metric/` Module:** Offers a comprehensive suite of evaluation metrics for time series tasks, managed by the `Evaluator` class. Includes standard metrics (MSE, MAE, etc.) and specialized sparse metrics for masked data.
*   **`visualize.py` Module:**  Equips users with visualization tools to plot time series data and model predictions, facilitating model analysis and interpretation through line charts and comparable real vs. predicted plots.
*   **`generative/` Module:** (Optional, if you want to highlight) Focuses on generative time series modeling, providing implementations of Time series VAEs and Transformer-based VAEs.

## Installation

Ensure you have Python installed. Then, to install pyFAST and its dependencies, run:

```bash
pip install -r requirements.txt
```

## Getting Started

### Basic Usage Example

Jumpstart your time series projects with pyFAST using this basic example:

```python
import torch
import torch.utils.data as data

from fast import initial_seed, get_device
from fast.data import SSTDataset
from fast.train import Trainer, to_string
from fast.metric import Evaluator
from fast.model.mts.ar import ANN  # Example: Using a simple ANN model

# Initialize components for reproducibility and evaluation
initial_seed(2025)

# Prepare your time series data: replace with actual data loading.
ts = torch.sin(torch.arange(0, 100, 0.1)).unsqueeze(1)  # Shape: (1000, 1)
train_ds = SSTDataset(ts, input_window_size=10, output_window_size=1).split(0.8, 'train')
val_ds = SSTDataset(ts, input_window_size=10, output_window_size=1).split(0.8, 'val')

# Initialize the model (e.g., ANN)
model = ANN(
    input_window_size=train_ds.input_window_size,  # Adapt input window size from dataset
    input_vars=train_ds.input_vars,  # Adapt input variable number from dataset
    output_window_size=train_ds.output_window_size,  # Adapt output window size from dataset, a.k.a. prediction steps
    hidden_size=32  # Hidden layer size
)

# Set up the Trainer for model training and evaluation
device = get_device('cpu')  # Use 'cuda', 'cpu', or 'mps'
evaluator = Evaluator(['MAE', 'RMSE'])  # Evaluation metrics

trainer = Trainer(device, model, evaluator=evaluator)

# Train model using prepared datasets
trainer.fit(train_ds, val_ds, epoch_range=(1, 10))  # Train for 10 epochs

# After training, evaluate on a test dataset (if available)
y_hat, y = trainer.predict(data.DataLoader(val_ds), 'evaluate val ')
loss = trainer.criterion(y_hat, *y)
metric_dict = trainer.evaluator.evaluate(y_hat, *y)
print(to_string('val: ', loss, *metric_dict.values()))
```

### Data Structures Overview

pyFAST is designed to handle various time series data structures:

*   **Multiple Time Series (MTS):**
    *   Shape: `[batch_size, window_size, n_vars]`
    *   For datasets with multiple variables recorded over time (e.g., sensor readings, stock prices of multiple companies).

*   **Univariate Time Series (UTS):**
    *   Shape: `[batch_size * n_vars, window_size, 1]`
    *   For datasets focusing on single-variable sequences, often processed in batches for efficiency.

*   **Advanced Data Handling:**
    *   **Sparse Data Ready:**  Models and metrics are designed to effectively work with sparse time series data and missing values, utilizing masks for accurate computations.
    *   **Exogenous Variable Integration:** Seamlessly incorporate external factors (exogenous variables) to enrich your time series models.
    *   **Variable-Length Sequence Support:**  Utilizes dynamic padding to efficiently process time series with varying lengths within batches, optimizing training and inference.

## Benchmarking Performance

pyFAST's performance and efficiency have been rigorously evaluated against established time series libraries and models on benchmark datasets.

**Benchmarking Highlights:**

*   **Datasets:** Evaluated on ETT-small (long-term forecasting), Electricity Load, and XMC-DC (real-world outpatient) datasets.
*   **Baselines:** Compared against strong baselines including Informer, PatchTST, and GluonTS (DeepAR, Transformer).
*   **Metrics:** Performance assessed using MSE, MAE, RMSE, and MAPE.
*   **Results:**  pyFAST models, especially Transformer-based architectures, demonstrate **competitive or superior performance** in forecasting accuracy while maintaining **computational efficiency** due to optimized implementations and modular design.


### Benchmarking on Univariate/Multivariate Time Series Datasets
**Example Benchmarking Results (Table 1 from Paper):**

| Model                     | Dataset          |    MSE |   MAE |  RMSE |  MAPE |
|---------------------------|------------------|-------:|------:|------:|------:|
| pyFAST (Transformer)      | ETT-small        |  0.123 | 0.087 | 0.351 | 0.054 |
| Informer                  | ETT-small        |  0.135 | 0.092 | 0.367 | 0.058 |
| PatchTST                  | ETT-small        |  0.128 | 0.090 | 0.358 | 0.056 |
| GluonTS (Transformer)     | ETT-small        |  0.140 | 0.095 | 0.374 | 0.060 |
| pyFAST (Transformer)      | Electricity Load |  0.085 | 0.063 | 0.292 | 0.041 |
| GluonTS (DeepAR)          | Electricity Load |  0.092 | 0.068 | 0.303 | 0.045 |
| pyFAST (GNN)              | XMC-DC           |  0.057 | 0.042 | 0.239 | 0.032 |
| LSTM                      | XMC-DC           |  0.065 | 0.048 | 0.255 | 0.036 |

### Benchmarking on Sparse Time Series Datasets

Protein dataset

### Benchmarking on Fusible Time Series Datasets



## License

MIT License

Copyright (c) 2024 pyFAST Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
