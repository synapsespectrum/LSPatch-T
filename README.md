# LSPatch-T: Long-Short Patch Transferring for Multivariate Time Series Forecasting

🌟 LSPatch-T introduces a novel framework that bridges short-term and long-term temporal patterns by transferring knowledge from short-length patch tokens to full-length variate tokens. Transfer learning is the key to effective MTSF.

🏆 LSPatch-T achieves state-of-the-art performance across multiple datasets while maintaining efficiency and interpretability in multivariate time series forecasting tasks.

The repo is the official implementation for the paper: 
[Long-Short Patch Transferring for Multivariate Time Series Forecasting]()
## Key Features

- Two-Phase Learning Framework: Pretraining and downstream fine-tuning with full-length variate tokens
- Channel Independence Design
- Robust Frequency Loss to scale the variational loss
![](imgs/frequency.png)
- Transformer-MLP Architecture: Combines Transformer encoder for pretraining and MLP blocks for downstream tasks

## Architecture Overview
![ LSPatch-T self-supervised framework](imgs/LSPatch-T-architecture.png)


## Reproducibility

To easily reproduce the results you can follow the next steps:
1. Clone the repository: 
```bash
# Clone the repository
git clone https://github.com/synapsespectrum/LSPatch-T.git
cd LSPatch-T

# Install dependencies
pip install -r requirements.txt
```
2. We're using 3 public datasets: ETT (ETTh1, ETTh2, ETTm1, ETTm2), ECL and Weather. Download these datasets using: 
```
make dataset
```
Expected folder structure [Structure](structure.md).
3. Train and Evaluate: We provide comprehensive scripts for different scenarios
```bash
# Pretrain and fine-tune LSPatch-T on ECL dataset
bash ./scripts/experiments/ECL/LSPatchT.sh

# Cross-dataset evaluation on ETT dataset
bash ./scripts/ablation/transferlearning/etth2-etth1.sh

# 
```


## MLFlow Integration

### How to Integrate MLFlow into Your Code

1. **Install MLFlow**:
   Ensure you have MLFlow installed in your environment. You can install it using pip:
``` shell
pip install mlflow
```

2. **Import MLFlow: Import MLFlow in your Python script:**  
``` python
import mlflow
import mlflow.pytorch
```
3. **Set Up MLFlow Experiment**:  
``` python
mlflow.set_experiment("experiment_name")
Start an MLFlow Run: Start an MLFlow run to log parameters, metrics, and artifacts:  
with mlflow.start_run(run_name="run_name") as run:
    mlflow.log_param("param_name", param_value)
    mlflow.log_metric("metric_name", metric_value)
    mlflow.log_artifact("path/to/artifact")
```


### Metrics Tracked
The following metrics are tracked during the training and evaluation of the model:  

- Training Metrics:  
  - `train_loss`: The training loss for each epoch.
  - `batch_loss`: The loss for each batch.
  - `batch_speed`: The speed of processing each batch.
  - grad`ient_norm`: The norm of the gradients.
- Validation Metrics:  
  - `vali_loss`: The validation loss for each epoch.
  - `best_vali_loss`: The best validation loss achieved.
- Test Metrics:  
  - `test_loss`: The test loss for each epoch.
  - `mae`: Mean Absolute Error.
  - `mse`: Mean Squared Error.
  - `rmse`: Root Mean Squared Error.
  - `mape`: Mean Absolute Percentage Error.
  - `mspe`: Mean Squared Percentage Error.
- System Metrics:  
  - `model_size_mb`: The size of the model in megabytes.
  - `epoch_time`: The time taken for each epoch.
  - `vali_time`: The time taken for validation.
  - `test_time`: The time taken for testing.

### Starting the MLFlow Server
1. Run the Server:  `sh server_manager.sh baseline start`
2. Check Server Status: 
`sh server_manager.sh baseline status`
3. Stop the Server: 
`sh server_manager.sh baseline stop`
4. Access the MLFlow UI: Open your web browser and go to `http://127.0.0.1:8080` to access the MLFlow UI and view your logged experiments and metrics.  

### Citation
```
@article{LSPatchT2024,
  title={Transferring from short to long sub-series for multivairate time series forecasting},
  author={[Le Hoang Anh]},
  journal={Energies},
  year={2024}
}
```

### Copyright @2024 Andrew Lee