# FAST: Forecasting and Time-Series in PyTorch

FAST is a comprehensive framework for time series analysis, supporting forecasting, imputation, and generation tasks. Its modular design enables flexible component composition and analysis, allowing users to easily combine and customize different modules for specific time series applications. The framework excels in model fusion, supporting seamless integration of different architectures, data types, and learning paradigms to create powerful hybrid solutions for complex time series challenges.

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

[Add Citation Information]
