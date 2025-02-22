# FAST: Forecasting and time-series in PyTorch

A comprehensive framework for time series analysis, supporting forecasting, imputation, and generation tasks.

## Features

- Multiple time series architectures (MTS/UTS)
- Flexible data preprocessing and fusion
- Cross-dataset training support
- Comprehensive evaluation metrics
- Multi-device acceleration (CPU/GPU/MPS)

## Installation

```bash
pip install -r requirements.txt
```

## Examples

### Basic Usage

```python
from fast import initial_seed
from fast.data import StandardScale
from fast.train import Trainer
from fast.metric import Evaluator

# Initialize components
initial_seed(42)
scaler = StandardScale()
evaluator = Evaluator(['MAE', 'RMSE', 'MAPE'])

# Train model
trainer = Trainer(device='cuda', model=model, evaluator=evaluator)
trainer.fit(train_ds, val_ds)
```

### Data Structures

1. Multiple Time Series (MTS)
```python
x -> [batch_size, window_size, n_vars]
```

2. Univariate Time Series (UTS) 
```python
x -> [batch_size * n_vars, window_size, 1]
```

3. Data Fusion Support
- Missing data handling with masks
- Exogenous variable integration
- Variable-length sequence support

## Coding Style

### Type Annotations
```python
def forecast(x: torch.Tensor, horizon: int = 1) -> torch.Tensor:
    """
    Forecast future values.
    
    Args:
        x: Input tensor [batch_size, window_size, n_vars]
        horizon: Forecast horizon
    
    Returns:
        Predicted values [batch_size, horizon, n_vars]
    """
    pass
```

### Documentation Guidelines
- Clear function descriptions
- Parameter specifications
- Return value details
- Usage examples for complex functions

## Research Topics

### 1. Generative Learning
- Cross-granularity estimation
  - Fine to coarse-grained data estimation
  - Coarse to fine-grained data estimation
- Applications in disease surveillance

### 2. Normalization Studies
- Instance vs Global Scale Analysis
  - Instance scale: Better for inference
  - Global scale: Better for generation
  - Scaling strategies for growing datasets

### 3. Data Fusion Techniques
- Missing data handling
- Exogenous variable integration
- Multi-source data fusion

## Ongoing Projects

### Healthcare Analytics

1. Glucose Monitoring (v1)
- Short-term sequence forecasting
- Cross-dataset training [sh_diabetes, kdd2018_glucose]
- Metrics: MAE, MAPE, RMSE, PCC
- Focus: Model training mechanism optimization

2. Glucose Estimation (v2)
- Generative learning approach
- Datasets: sh_diabetes, mimic-iii
- Food intake correlation analysis
- Metrics: MAE, MAPE, RMSE, PCC

### Energy Systems

1. Battery Health Estimation
- Generative pretrained modeling
- Downstream task optimization
- Metrics: MAE, MAPE, SDRE, PCC
- Impact analysis:
  - Annual battery capacity
  - Industry usage patterns
  - Recycling efficiency
  - Environmental impact

2. Wind Power Forecasting
- Physics-informed neural networks
- ODE/PDE integration
- Datasets:
  - la-haute-borne (500% improvement)
  - KDD2022-SDWPF
- Physical information integration

## Development Notes

### Data Structure Evolution
1. MTS Framework
- Tensor shape: [batch_size, window_size, n_vars]
- Deep reinforcement learning integration

2. UTS Framework
- Tensor shape: [batch_size * n_vars, window_size, 1]
- Variable length support
- Cross-dataset learning rate adaptation

3. Fusion Capabilities
- Mask support for missing data
- Exogenous time series integration
- Combined mask handling

### Embedding Research
- Time series variable embedding analysis
- Comparison with LLM embeddings
- Performance analysis:
  - MTS vs UTS approach
  - Case study: XMCDC disease data

## License

[Add License Information]

## Citation

[Add Citation Information]
```