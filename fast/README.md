# fast directory in pyFAST

This directory (`fast/`) is a core module within the pyFAST framework, focusing on efficient and modular implementations for time series analysis. It provides functionalities for data handling, model training, evaluation, and visualization, specifically designed for both univariate and multivariate time series data.

## Subdirectories and Files

* **`data/`**: This subdirectory is dedicated to data-related utilities.
    * **`__init__.py`**: Initializes the `data` package.
    * **`bdp_dataset.py`**: Likely contains implementation for BDP (Batch Data Provider) datasets, potentially for handling large datasets in batches.
    * **`mtm_dataset.py`**: Likely contains implementation for MTM (Multi-Task Learning) datasets, possibly for datasets used in multi-task learning scenarios.
    * **`patch.py`**: Implements patch-related functionalities, potentially for patch-based time series processing, which can be useful for models like PatchTST.
    * **`scale.py`**: Provides various scaling methods for time series data normalization, including `Scale`, `MinMaxScale`, `MeanScale`, `MaxScale`, `StandardScale`, `LogScale`, `InstanceScale`, and `InstanceStandardScale`. It also includes a `time_series_scaler` function for applying these scalers.
    * **`stm_dataset.py`**: Likely contains implementation for STM (Spatio-Temporal Modeling) datasets.
    * **`sts_dataset.py`**: Likely contains implementation for STS (Single Time Series) datasets, and includes utility functions like `multi_step_ahead_split` and `train_test_split` for time series data preprocessing.

* **`generative/`**: This subdirectory focuses on generative models for time series.
    * **`__init__.py`**: Initializes the `generative` package.
    * **`transvae.py`**: Likely implements a Transformer-based Variational Autoencoder (VAE) for generative time series modeling.
    * **`tvae.py`**: Likely implements a Time series Variational Autoencoder (VAE) for generative time series modeling.

* **`metric/`**: This subdirectory provides tools for evaluating time series models.
    * **`__init__.py`**: Initializes the `metric` package.
    * **`evaluate.py`**: Implements the `Evaluator` class, which is used to calculate and manage different evaluation metrics.
    * **`mask_metric.py`**: Defines metrics specifically for masked time series data, such as `mask_mean_absolute_error`, `mask_mean_squared_error`, etc., useful for handling missing values.
    * **`metric.py`**: Implements a wide range of common evaluation metrics for time series forecasting and analysis, including MSE, RMSE, MAE, MAPE, SMAPE, R², and more.

* **`model/`**: This subdirectory houses the model implementations.
    * **`__init__.py`**: Initializes the `model` package.
    * **`base/`**: Contains base modules and components used to build more complex models.
        * **`__init__.py`**: Initializes the `model.base` package and exports various base modules like `get_activation_cls`, `MLP`, `DirectionalRepresentation`, attention mechanisms (`SelfAttention`, `SymmetricAttention`, `MultiHeadSymmetricAttention`), and utility functions (`rolling_forecasting`, `count_parameters`, `freeze_parameters`, `covert_parameters`, `init_weights`, `to_string`).
        * **`activation.py`**: Defines different activation functions that can be used in models.
        * **`attention.py`**: Implements various attention mechanisms, which are crucial for many modern time series models, especially transformer-based ones.
        * **`decomposition.py`**: Likely contains modules for time series decomposition techniques.
        * **`dr.py`**: Likely defines Directional Representation layers, as suggested by `DirectionalRepresentation` in `base/__init__.py`.
        * **`gadgets.py`**: May contain miscellaneous utility modules or layers.
        * **`mlp.py`**: Implements Multi-Layer Perceptron (MLP) modules.
        * **`shapelet.py`**: Likely implements shapelet-related layers, possibly for shapelet-based time series analysis models.
        * **`utils.py`**: Provides utility functions for model building and manipulation.
    * **`modeler.py`**: Might contain a high-level class or functions to manage and instantiate different models within the `fast` framework.
    * **`mts/`**: Intended for Multiple Time Series (MTS) models (though currently seems empty based on the folder content).
    * **`uts/`**: Intended for Univariate Time Series (UTS) models (though currently seems empty based on the folder content).

* **`train.py`**: Implements the `Trainer` class and `EarlyStop` class, which are essential for training time series models.
    * **`Trainer`**:  Provides a comprehensive training loop with functionalities for fitting models, prediction, evaluation, checkpoint saving, early stopping, and device management (CPU, CUDA, MPS). It supports features like learning rate scheduling, data scaling, and handling of additive loss criteria.
    * **`EarlyStop`**: Implements early stopping mechanism to prevent overfitting by monitoring a validation loss and stopping training when it no longer improves.

* **`visualize.py`**: Provides visualization tools for time series data and model predictions using `matplotlib`.
    * Includes functions like `plot_in_line_chart` for plotting equal-length time series, `plot_jagged_ts_in_line_chart` for unequal-length time series, and `plot_comparable_line_charts` for comparing real and predicted time series.

* **`__init__.py`**: Initializes the `fast` package, sets the version, and includes utility functions.
    * **`initial_seed`**: Function to fix random seeds for reproducibility across `random`, `numpy`, and `torch`.
    * **`get_common_params`**: Function to extract common parameters between a function's signature and a given parameter dictionary.


This directory structure and the provided files suggest that `pyFAST/fast` is a well-organized and feature-rich module designed to support a wide range of time series analysis tasks, with a strong emphasis on modularity and flexibility. It covers data handling, model building, training, evaluation, and visualization, making it a comprehensive toolkit for time series practitioners and researchers.
