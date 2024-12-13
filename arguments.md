## 🛠️ Arguments

#### Basic Configurations
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--is_training` | int | 1 | Whether to train the model |
| `--is_pretrain` | int | 0 | Enable pretraining mode |
| `--is_finetune` | int | 0 | Enable fine-tuning mode |
| `--pretrained_model` | str | None | Path to pretrained model |
| `--model_id` | str | 'test' | Identifier for the experiment |
| `--model` | str | 'Transformer' | Model type: [Transformer, Informer, Autoformer, PatchTST, LSPatchT, iTransformer, Crossformer] |
| `--use_mlflow` | bool | True | Enable MLflow tracking |

#### Data Loading
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | 'ETTh1' | Dataset name |
| `--root_path` | str | './data/ETT/' | Root directory for data |
| `--data_path` | str | 'ETTh1.csv' | Specific data file |
| `--target` | str | 'OT' | Target feature for prediction |
| `--freq` | str | 'h' | Time feature encoding: [s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly] |
| `--checkpoints` | str | './experiments/model_saved/checkpoints/' | Directory for model checkpoints |

#### Forecasting Configuration
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--features` | str | 'M' | Task type: [M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate] |
| `--seq_len` | int | 96 | Input sequence length |
| `--label_len` | int | 48 | Start token length |
| `--pred_len` | int | 96 | Prediction sequence length |

#### Model Architecture
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--enc_in` | int | 7 | Encoder input size |
| `--dec_in` | int | 7 | Decoder input size |
| `--c_out` | int | 7 | Output size |
| `--d_model` | int | 512 | Model dimension |
| `--n_heads` | int | 8 | Number of attention heads |
| `--e_layers` | int | 2 | Number of encoder layers |
| `--d_layers` | int | 1 | Number of decoder layers |
| `--d_ff` | int | 2048 | Dimension of FCN |
| `--moving_avg` | int | 25 | Moving average window size |
| `--factor` | int | 1 | Attention factor |
| `--distil` | bool | True | Use distilling in encoder |
| `--dropout` | float | 0.05 | Dropout rate |
| `--embed` | str | 'timeF' | Time features encoding: [timeF, fixed, learned] |
| `--mask_ratio` | float | 0.4 | Masking ratio for input |
| `--patch_len` | int | 12 | Patch length |
| `--stride` | int | 12 | Stride between patches |

#### Optimization
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--num_workers` | int | 10 | Number of data loading workers |
| `--itr` | int | 2 | Number of experiment iterations |
| `--train_epochs` | int | 10 | Number of training epochs |
| `--batch_size` | int | 32 | Training batch size |
| `--patience` | int | 3 | Early stopping patience |
| `--learning_rate` | float | 0.0001 | Learning rate |
| `--loss` | str | 'mse' | Loss function |
| `--lradj` | str | 'type1' | Learning rate adjustment type |
| `--use_amp` | bool | False | Use automatic mixed precision training |

#### GPU Settings
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_gpu` | bool | True | Enable GPU usage |
| `--gpu` | int | 0 | GPU device ID |
| `--use_multi_gpu` | bool | False | Enable multi-GPU training |
| `--devices` | str | '0,1,2,3' | Device IDs for multi-GPU training |