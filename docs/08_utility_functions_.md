# Chapter 8: Utility Functions

Welcome back! In the previous chapters, we've unpacked the main parts of the LSPatch-T project: the [Model Architectures](01_model_architectures_.md) (the blueprints), the [Experiment Runner](02_experiment_runner_.md) (the conductor), the [Data Providers](03_data_providers_.md) (the data librarians), the [Training and Evaluation Logic](04_training_and_evaluation_logic_.md) (the learning process), the [Loss Functions](05_loss_functions_.md) (the grading system), and the [Core Layers](06_core_layers_.md) (the building blocks). We even looked at [MLflow Tracking](07_mlflow_tracking_.md) for keeping your experiments organized.

While these are the main "engines" and "framework" of the project, any complex software also needs many smaller, helpful tools to get specific jobs done efficiently.

This is where **Utility Functions** come in.

### What are Utility Functions?

Imagine you're working on a big project, like building something or doing complex research. You have your main tools (like hammers, saws, or microscopes), but you also need lots of smaller, handy gadgets and helpers:

*   A calculator to quickly do sums.
*   A ruler to measure precisely.
*   Graph paper to plot results.
*   A dictionary to look up terms.

You don't build these helper tools *into* your main equipment; you keep them separate and use them whenever you need them.

In the LSPatch-T project, Utility Functions are like these handy tools and gadgets. They are helper pieces of code that perform specific, often repetitive, tasks that support the main workflow but aren't core parts of the model's logic, data reading, or training loops themselves.

They are typically found in the `utils` directory:

```
LSPatch-T/
├── utils/
│   ├── RevIN.py         # For data normalization
│   ├── metrics.py       # For calculating evaluation scores
│   ├── timefeatures.py  # For encoding dates/times into numbers
│   └── tools.py         # Various helpers (early stopping, learning rate, plotting, etc.)
├── ... other directories and files
```

These files contain functions and small classes that the main components of the project (like the Experiment Runner or Data Providers) call upon when they need a specific task performed.

### Your First Use Case: Preparing Data and Measuring Performance

You interact with utility functions indirectly. When you run an experiment using `experiment.py`, the [Experiment Runner](02_experiment_runner_.md) and the [Data Providers](03_data_providers_.md) automatically use these utilities behind the scenes.

Two very common tasks that rely on utility functions are:

1.  **Adding time information to data:** Your raw data might just have values and timestamps. The model might need numerical features representing the hour of the day, day of the week, etc., to understand patterns. A utility function handles this conversion.
2.  **Calculating final evaluation scores:** After testing, you need to know how well your model performed using standard metrics like MAE and MSE. A utility function calculates these scores from the model's predictions and the true values.

Let's look at some specific examples from the `utils` directory.

### Adding Time Features (`utils/timefeatures.py`)

As mentioned, time series models often benefit from having numerical features derived from the timestamp (like month, day, hour). This helps them capture seasonality and cyclical patterns.

The `utils/timefeatures.py` file contains functions and classes designed to take a list of dates or timestamps and generate these numerical features.

**How it's used:**

The [Data Providers](03_data_providers_.md), specifically the `Dataset` classes (like `Dataset_ETT_hour`) in `data_provider/data_loader.py`, call a function from `timefeatures.py` when they are reading and processing the data.

```python
# Inside data_provider/data_loader.py (simplified snippet from __read_data__)
import pandas as pd
from utils.timefeatures import time_features # Import the utility function

def __read_data__(self):
    # ... (read CSV into df_raw) ...

    # Extract date column for the current data split (train/val/test)
    df_stamp = df_raw[['date']][border1:border2]
    df_stamp['date'] = pd.to_datetime(df_stamp.date) # Ensure it's datetime objects

    # --- Use the utility function to generate time features ---
    # Calls the time_features utility with the dates, encoding type, and frequency
    data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

    # Store the generated time features
    self.data_stamp = data_stamp # This will be used by __getitem__

    # print(f"Generated time features with shape: {data_stamp.shape}") # Example output: (num_samples, num_time_features)
```
**Explanation:** The `Dataset` class calls the `time_features` function, passing the DataFrame column containing the dates. The `time_features` function then calculates numerical representations (like scaled hour of day, day of week, etc.) and returns them as a NumPy array or similar structure, which the Dataset stores and provides alongside the actual time series values.

**How it works (simplified):**

The `time_features` function inside `utils/timefeatures.py` looks at the `timeenc` parameter and the `freq` (frequency, e.g., 'h' for hour). Based on these, it selects a set of helper classes (like `HourOfDay`, `DayOfWeek`, `MonthOfYear`, etc.) defined in the same file. It then applies these classes to the datetime objects to extract and format the desired numerical features.

```python
# Inside utils/timefeatures.py (simplified snippet)
import numpy as np
import pandas as pd

# Example of one TimeFeature class
class HourOfDay:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # Extract the hour (0-23) and scale it between -0.5 and 0.5
        return index.hour / 23.0 - 0.5

# Simplified time_features function
def time_features(dates, timeenc=1, freq='h'):
    # Ensure input dates are datetime objects
    dates = pd.to_datetime(dates.date.values)

    if timeenc == 1:
        # Based on freq, select relevant TimeFeature classes
        if freq == 'h':
            # For hourly data, include HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
            features_list = [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
        elif freq == 'd':
             # For daily data, maybe just DayOfWeek, DayOfMonth, DayOfYear
             features_list = [DayOfWeek(), DayOfMonth(), DayOfYear()]
        # ... (more frequencies) ...
        else:
             # Default or error
             raise NotImplementedError

        # Apply each selected TimeFeature to the dates
        # np.vstack stacks the results as rows, transpose(1,0) makes them columns
        return np.vstack([feat(dates) for feat in features_list]).transpose(1,0)

    # ... (timeenc == 0 logic) ...
```
**Explanation:** The `time_features` function acts as a dispatcher. It uses helper classes (`HourOfDay`, etc.) that each know how to calculate a single type of time feature. It gathers the results from all selected features and combines them into a single array where each row corresponds to a timestamp and each column is a different time feature.

### Data Normalization (`utils/RevIN.py`, `utils/tools.py`)

Neural networks often perform better when the input data is scaled or normalized so that all values are within a similar range (e.g., around zero). This prevents features with larger values from dominating the learning process.

The project uses a couple of approaches for this:

1.  **`StandardScaler`:** A simple utility class in `utils/tools.py` that calculates the mean and standard deviation of training data and uses these to scale data by subtracting the mean and dividing by the standard deviation.
2.  **`RevIN` (Reversible Instance Normalization):** A more advanced technique implemented as a PyTorch layer in `utils/RevIN.py`. It normalizes data based on the statistics of the *current batch* or sequence and includes a "denormalization" step to reverse the process after the model makes a prediction. This is particularly useful for time series.

**How it's used (`StandardScaler`):**

`StandardScaler` is typically used by the [Data Providers](03_data_providers_.md) during the `__read_data__` phase to normalize the entire dataset split (train/val/test) *after* calculating statistics *only* on the training portion.

```python
# Inside data_provider/data_loader.py (simplified snippet from __read_data__)
from sklearn.preprocessing import StandardScaler # Standard library scaler (often used)
from utils.tools import StandardScaler # Or the custom one if used

def __read_data__(self):
    # ... (read data into df_data) ...

    if self.scale:
        # Fit the scaler ONLY on the training data
        train_data = df_data[border1s[0]:border2s[0]]
        # self.scaler = StandardScaler() # Use sklearn's
        self.scaler = StandardScaler(mean=train_data.values.mean(axis=0), std=train_data.values.std(axis=0)) # Or custom one
        # self.scaler.fit(train_data.values) # If using sklearn

        # --- Use the utility scaler to transform data ---
        # Apply the transformation to the current data split
        data = self.scaler.transform(df_data.values)

    # Store the normalized data
    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data = data # Keep full scaled data for inverse_transform

    # print(f"Data scaled successfully.")
```
**Explanation:** The `Dataset` class creates an instance of `StandardScaler`, fits it to the training data, and then uses its `transform` method to scale the data for the current train/val/test split. It also keeps the scaler instance to be able to reverse the scaling (`inverse_transform`) later in the Experiment Runner after predictions are made.

**How it's used (`RevIN`):**

`RevIN` is often used as a layer *within* the model architecture itself, applied just after the input data arrives and just before the final output.

```python
# Inside models/PatchTST.py (simplified Model __init__ and forward)
import torch.nn as nn
from utils.RevIN import RevIN # Import the RevIN layer utility

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # ... define other layers ...

        # --- Add RevIN layer ---
        self.revin_layer = RevIN(configs.enc_in) # Create an instance for 'enc_in' features

        # ... define embedding, encoder, head ...

    def forward(self, x_enc, x_mark_enc, ...):
        # x_enc is [Batch, seq_len, n_vars]

        # --- Use RevIN to normalize the input ---
        x_enc = self.revin_layer(x_enc, 'norm') # Apply normalization mode

        # ... data flows through embedding, encoder, head ...
        # result from head is [Batch, pred_len, n_vars]

        # --- Use RevIN to denormalize the output ---
        x = self.revin_layer(x, 'denorm') # Apply denormalization mode

        return x
```
**Explanation:** The model's `__init__` creates an instance of the `RevIN` class. In the `forward` method, the input `x_enc` is passed through `self.revin_layer` with `mode='norm'` to normalize it *before* any processing. The final prediction `x` is passed through the *same* `self.revin_layer` with `mode='denorm'` to convert it back to the original scale *before* returning.

**How it works (`RevIN` simplified):**

The `RevIN` class in `utils/RevIN.py` stores the mean and standard deviation of the input data when in `'norm'` mode. It then applies the standard normalization formula `(x - mean) / stdev`. In `'denorm'` mode, it reverses this formula `(x * stdev) + mean` using the stored statistics. It also includes optional learnable affine parameters (`affine_weight`, `affine_bias`) to further enhance the transformation.

```python
# Inside utils/RevIN.py (simplified)
import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features)) # Learnable weights
            self.affine_bias = nn.Parameter(torch.zeros(num_features)) # Learnable biases

    def forward(self, x, mode:str):
        if mode == 'norm':
            # Calculate and store mean/stdev for the current input batch/sequence
            dim2reduce = tuple(range(1, x.ndim-1)) # Dimensions to average over (e.g., time steps)
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
            # Apply normalization
            x = x - self.mean
            x = x / self.stdev
            if self.affine: # Apply learnable affine transformation
                x = x * self.affine_weight
                x = x + self.affine_bias
        elif mode == 'denorm':
             if self.affine: # Reverse affine transformation
                x = x - self.affine_bias
                x = x / (self.affine_weight + self.eps*self.eps) # Add eps^2 for stability before division
             # Apply denormalization using stored mean/stdev
             x = x * self.stdev
             x = x + self.mean
        else: raise NotImplementedError
        return x
```
**Explanation:** `RevIN` is stateful; it remembers the mean and standard deviation from the `'norm'` pass to use them for the `'denorm'` pass. This ensures the denormalization perfectly reverses the normalization for that specific input, making the transformation "reversible".

### Evaluation Metrics (`utils/metrics.py`)

To assess the final performance of a trained model on the test set, standard evaluation metrics are used. `utils/metrics.py` provides functions to calculate common forecasting metrics like Mean Absolute Error (MAE) and Mean Squared Error (MSE).

**How it's used:**

The [Experiment Runner](02_experiment_runner_.md), specifically the `test()` method of `Exp_Main`, collects the model's predictions and the true values for the entire test set. It then calls the `metric` function from `utils/metrics.py` to get the final scores.

```python
# Inside exp/exp_main.py (simplified snippet from test method)
import numpy as np
from utils.metrics import metric # Import the utility function

def test(self, setting, test=0):
    # ... (Run model on test data, collect predictions and true values) ...
    # Assume 'preds' is a numpy array of predictions
    # Assume 'trues' is a numpy array of corresponding true values

    # --- Use the utility function to calculate metrics ---
    # Call the metric function with predictions and true values
    mae, mse, rmse, mape, mspe = metric(preds, trues)

    print(f'Test set: MSE:{mse:.7f}, MAE:{mae:.7f}')

    # ... (Log metrics, save results) ...
```
**Explanation:** The `test()` method simply calls the `metric` function, passing the collected test predictions and true values. The utility function does the calculations and returns the scores.

**How it works (simplified):**

The `metric` function in `utils/metrics.py` itself calls other simpler functions within the same file (`MAE`, `MSE`, `RMSE`, etc.), each of which calculates one specific metric using standard formulas on the input arrays.

```python
# Inside utils/metrics.py (simplified snippet)
import numpy as np

# Function to calculate Mean Absolute Error
def MAE(pred, true):
    # Calculate absolute differences, then take the mean
    return np.mean(np.abs(pred - true))

# Function to calculate Mean Squared Error
def MSE(pred, true):
     # Calculate squared differences, then take the mean
    return np.mean((pred - true)**2)

# The main metric function that calls the others
def metric(pred, true):
    # Call individual metric functions
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = np.sqrt(mse) # RMSE is just sqrt of MSE
    # ... calculate MAPE, MSPE ...

    # Return all calculated metrics
    return mae, mse, rmse, None, None # Simplified return
```
**Explanation:** This file contains basic mathematical functions implemented using NumPy, performing array operations according to the definitions of each metric.

### Other Useful Tools (`utils/tools.py`)

The `utils/tools.py` file is a collection of miscellaneous but helpful functions used by the Experiment Runner:

*   **`EarlyStopping`:** A class that monitors validation loss and stops training early if it plateaus. Used in the `train()` method ([Chapter 4: Training and Evaluation Logic](04_training_and_evaluation_logic_.md)).
*   **`adjust_learning_rate`:** A function to modify the optimizer's learning rate during training. Used in the `train()` method.
*   **`visual`:** A simple function to plot predicted vs. true values for visualization. Can be used after testing.
*   **`dotdict`:** A small class that allows accessing dictionary keys using dot notation (e.g., `args.model` instead of `args['model']`), making the code cleaner.
*   **`StandardScaler`:** As discussed, a utility for basic data scaling.
*   **`get_model_size`:** A function to calculate the memory footprint of a PyTorch model.
*   **`read_yaml_file`:** A helper to read configuration from YAML files.

These utilities are called at appropriate points by the Experiment Runner during the experiment lifecycle, as outlined in [Chapter 4: Training and Evaluation Logic](04_training_and_evaluation_logic_.md).

### Summary of Key Utility Types

| Utility File/Class        | Main Purpose                                       | Used By (Examples)                          | Key Functions/Classes Illustrated Here |
| :------------------------ | :------------------------------------------------- | :------------------------------------------ | :------------------------------------- |
| `utils/timefeatures.py`   | Generate numerical features from timestamps        | [Data Providers](03_data_providers_.md)     | `time_features`, `HourOfDay`           |
| `utils/RevIN.py`          | Reversible Instance Normalization (as a Layer)     | [Model Architectures](01_model_architectures_.md) | `RevIN`                                |
| `utils/tools.py`          | Various helpers: scaling, early stopping, LR, etc. | [Data Providers](03_data_providers_.md), [Experiment Runner](02_experiment_runner_.md) | `StandardScaler`, `EarlyStopping`, `adjust_learning_rate` |
| `utils/metrics.py`        | Calculate standard evaluation metrics              | [Experiment Runner](02_experiment_runner_.md)'s `test()` | `metric`, `MAE`, `MSE`                 |

These are just a few examples; the `utils` directory contains other helper code. The key idea is that they provide specific, well-defined functionalities that support the main components without cluttering their core logic.

### Conclusion

Utility functions are the helpful tools and gadgets that make the LSPatch-T project more efficient and organized. Found primarily in the `utils` directory, they handle specific tasks like converting dates to numerical features (`timefeatures.py`), normalizing data (`RevIN.py`, `tools.py`), calculating performance metrics (`metrics.py`), and implementing experiment control mechanisms like early stopping and learning rate adjustment (`tools.py`). These utilities are called upon by the main components like [Data Providers](03_data_providers_.md), [Model Architectures](01_model_architectures_.md), and the [Experiment Runner](02_experiment_runner_.md) whenever their specific functionality is needed, keeping the core code focused and clean. Understanding these utilities helps you appreciate the supporting infrastructure that makes the project run smoothly.

This chapter concludes our detailed look at the main components of the LSPatch-T project. You now have a foundational understanding of its architecture, how experiments are run, how data is handled, how the model learns, and the various supporting utilities that tie it all together.
