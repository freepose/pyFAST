# FAST: Forecasting and Time-Series in PyTorch

FAST (Forecasting And time-Series in PyTorch) is a powerful, modular framework designed for advanced time series analysis. It provides a unified interface for forecasting, imputation, and generative modeling tasks, with specialized support for both multiple (MTS) and univariate (UTS) time series. The framework's core strength lies in its flexible architecture, which seamlessly integrates transformer-based models, variational autoencoders, and traditional approaches.

Key capabilities include:
- Efficient handling of complex time series data with support for missing values and variable-length sequences
- Advanced data fusion techniques for combining multiple data sources and exogenous variables
- Comprehensive model implementations ranging from basic architectures to state-of-the-art transformer models
- Robust training pipelines with customizable evaluation metrics and device-agnostic acceleration
- Built-in support for generative modeling and synthetic data generation

Built on PyTorch, FAST emphasizes both usability and extensibility, making it suitable for both research experiments and production deployments in domains such as healthcare analytics and energy forecasting.

## Features

- Multiple time series architectures (MTS/UTS)
- Flexible data preprocessing and module fusion
- Cross-dataset training support
- Comprehensive evaluation metrics
- Multi-device acceleration (CPU/GPU/MPS)

## Installation

To install the pyFAST library, ensure you have Python installed, then run the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

## Getting Started

### Basic Usage

Here's a quick example to get you started with FAST:

```python
from fast import initial_seed
from fast.data import StandardScale
from fast.train import Trainer
from fast.metric import Evaluator
from fast.model import YourModel  # Replace with the specific model you are using

# Initialize components
initial_seed(42)
scaler = StandardScale()
evaluator = Evaluator(['MAE', 'RMSE', 'MAPE'])

# Prepare your data
train_ds = ...  # Load or prepare your training dataset
val_ds = ...    # Load or prepare your validation dataset

# Initialize and train your model
model = YourModel(...)  # Initialize your model with necessary parameters
trainer = Trainer(device='cuda', model=model, evaluator=evaluator)
trainer.fit(train_ds, val_ds)
```

### Data Structures

1. **Multiple Time Series (MTS)**
   - Shape: `[batch_size, window_size, n_vars]`
   - Used for datasets with multiple variables over time.

2. **Univariate Time Series (UTS)**
   - Shape: `[batch_size * n_vars, window_size, 1]`
   - Used for single-variable sequences.

3. **Data Fusion Support**
   - Handles missing data with masks
   - Integrates exogenous variables
   - Supports variable-length sequences

## Core Directory Structure

```plaintext
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
└── utils/
    ├── config.py        # Configuration utilities
    ├── logging.py       # Logging utilities
    └── tools.py         # General utility functions
```

Each directory serves a specific purpose:

- **data/**: Handles all data-related operations
  - Implements data loading, preprocessing, and dataset management
  - Supports both MTS and UTS data formats
  - Provides tools for data fusion and scaling

- **model/**: Contains all model architectures
  - Separate implementations for MTS and UTS approaches
  - Includes transformer-based and other architectural variants
  - Provides base components for model development

- **generative/**: Focuses on generative modeling
  - Implements VAE and other generative approaches
  - Provides tools for synthetic data generation

- **train/**: Manages the training process
  - Implements training loops and optimization
  - Provides callback mechanisms for training control

- **metric/**: Handles evaluation and losses
  - Implements various evaluation metrics
  - Provides loss functions for different tasks

- **utils/**: Contains utility functions
  - Manages configuration and logging
  - Provides common helper functions

## License

MIT License

Copyright (c) 2024 pyFAST Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Citation

If you use FAST in your research, please cite our related works:

1. Wang, Z., Liu, H., Wu, S., Liu, N., Liu, X., Hu, Y., & Fu, Y. (2024). Explainable time-varying directional representations for photovoltaic power generation forecasting. *Journal of Cleaner Production*, 143056.

2. Huang, Y., Zhao, Y., Wang, Z., Liu, X., & Fu, Y. (2024). Sparse dynamic graph learning for district heat load forecasting. *Applied Energy*, 371, 123685.

3. Hu, Y., Liu, H., Wu, S., Zhao, Y., Wang, Z., & Liu, X. (2024). Temporal collaborative attention for wind power forecasting. *Applied Energy*, 357, 122502.

4. Huang, Y., Zhao, Y., Wang, Z., Liu, X., Liu, H., & Fu, Y. (2023). Explainable district heat load forecasting with active deep learning. *Applied Energy*, 350, 121753.

5. Wang, Z., Liu, X., Huang, Y., Zhang, P., & Fu, Y. (2023). A multivariate time series graph neural network for district heat load forecasting. *Energy*, 127911.
