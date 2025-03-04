# pyFAST: A Powerful and Modular Time Series Framework

[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

**pyFAST** (Forecasting And Time-Series in PyTorch) is a robust, modular framework for advanced time series analysis, built on PyTorch. It offers a unified interface for forecasting, imputation, and generative modeling tasks, with specialized support for both multiple (MTS) and univariate (UTS) time series. Its flexible architecture seamlessly integrates transformer-based models, variational autoencoders, and traditional approaches.

**pyFAST** emphasizes both usability and extensibility, making it suitable for research experiments and production deployments in domains such as healthcare analytics, energy forecasting, and beyond.

## Key Capabilities

* **Versatile Time Series Handling:**
  * Efficiently handles complex time series data.
  * Supports missing values and variable-length sequences.
  * Supports Multiple Time Series (MTS) and Univariate Time Series (UTS).
* **Advanced Data Fusion:** Combines multiple data sources and exogenous variables.
* **Comprehensive Model Library:**
  * Includes basic architectures and state-of-the-art transformer models.
  * Supports Generative Models.
* **Robust Training:**
  * Customizable evaluation metrics.
  * Device-agnostic acceleration (CPU/GPU/MPS).
* **Data Scaler:** Provides common scalers for data normalization.
* **Reproducibility:** Provide `initial_seed` to fix random seed for reproducibility.
* **Generative Modeling:** Built-in support for synthetic data generation.
* **Cross-Dataset Training:** Supports training the model on multiple datasets.

## Installation

To install pyFAST, ensure you have Python installed. Then, install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Getting Started

### Basic Usage

Here's a simple example to get you started:

```python
from fast import initial_seed
from fast.data.preprocessing.scaler import StandardScale
from fast.train.trainer import Trainer
from fast.metric.evaluator import Evaluator
from fast.model.base import YourModel  # Replace with your model

# Initialize components
initial_seed(42)
scaler = StandardScale()
evaluator = Evaluator(['MAE', 'RMSE', 'MAPE'])

# Prepare your data (replace with your dataset)
train_ds = ...  # Load or prepare your training dataset
val_ds = ...    # Load or prepare your validation dataset

# Initialize and train your model
model = YourModel(...)  # Initialize your model with necessary parameters
trainer = Trainer(device='cuda', model=model, evaluator=evaluator, global_scaler=scaler) # Use scaler
trainer.fit(train_ds, val_ds)
```

**Note:**
- Replace `YourModel` with the specific model you want to use (from `fast.model.base`).
- Replace the `...` placeholders for `train_ds` and `val_ds` with your actual data loading or preparation code.

## Data Structures

- **Multiple Time Series (MTS):** `[batch_size, window_size, n_vars]`
  - Used for datasets with multiple variables over time.
- **Univariate Time Series (UTS):** `[batch_size * n_vars, window_size, 1]`
  - Used for single-variable sequences.

### Data Fusion Support

- Handles missing data with masks.
- Integrates exogenous variables.
- Supports variable-length sequences.

## Core Directory Structure

```
fast/
├── data/
│   ├── loader/           # Data loading utilities
│   ├── preprocessing/    # Data preprocessing tools
│   │   ├── scaler.py    # Data scaling implementations
│   │   └── fusion.py    # Data fusion utilities
│   └── dataset.py       # Dataset implementations
├── model/
│   ├── mts/             # Multiple Time Series models
│   │   ├── transformer/ # Transformer-based architectures
│   │   └── base/       # Base MTS implementations
│   ├── uts/             # Univariate Time Series models
│   │   ├── transformer/ # Transformer-based architectures
│   │   └── base/       # Base UTS implementations
│   └── base/            # Base model components
├── generative/          # Generative model implementations
│   ├── transvae.py      # Transformer VAE implementation
│   └── base.py          # Base generative components
├── train/
│   ├── trainer.py       # Training loop implementations
│   └── callbacks.py     # Training callbacks
├── metric/
│   ├── evaluator.py     # Evaluation metrics implementation
│   └── losses.py        # Loss functions
├── __init__.py          # Package initialization
└── utils/
    ├── config.py        # Configuration utilities
    ├── logging.py       # Logging utilities
    └── tools.py         # General utility functions
```

### Directory Descriptions:

- **`data/`**: Manages all data-related operations.
  - Implements data loading, preprocessing, and dataset management.
  - Supports both MTS and UTS data formats.
  - Provides tools for data fusion and scaling.
- **`model/`**: Houses all model architectures.
  - Separate implementations for MTS and UTS.
  - Includes transformer-based and other model variants.
  - Provides base components for model development.
- **`generative/`**: Focuses on generative modeling.
  - Implements VAE and other generative approaches.
  - Provides tools for synthetic data generation.
- **`train/`**: Manages the training process.
  - Implements training loops and optimization.
  - Provides callback mechanisms for training control.
- **`metric/`**: Handles evaluation and losses.
  - Implements various evaluation metrics.
  - Provides loss functions for different tasks.
- **`utils/`**: Contains utility functions.
  - Manages configuration and logging.
  - Provides common helper functions.
- **`__init__.py`**: Package initialization.

## License

### MIT License

```
Copyright (c) 2024 pyFAST Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Citation

If you use pyFAST in your research, please cite our related works:

- Wang, Z., Liu, H., Wu, S., Liu, N., Liu, X., Hu, Y., & Fu, Y. (2024). Explainable time-varying directional representations for photovoltaic power generation forecasting. *Journal of Cleaner Production*, 143056.
- Huang, Y., Zhao, Y., Wang, Z., Liu, X., & Fu, Y. (2024). Sparse dynamic graph learning for district heat load forecasting. *Applied Energy*, 371, 123685.
- Hu, Y., Liu, H., Wu, S., Zhao, Y., Wang, Z., & Liu, X. (2024). Temporal collaborative attention for wind power forecasting. *Applied Energy*, 357, 122502.
- Huang, Y., Zhao, Y., Wang, Z., Liu, X., Liu, H., & Fu, Y. (2023). Explainable district heat load forecasting with active deep learning. *Applied Energy*, 350, 121753.
- Wang, Z., Liu, X., Huang, Y., Zhang, P., & Fu, Y. (2023). A multivariate time series graph neural network for district heat load forecasting. *Energy*, 127911.
```

