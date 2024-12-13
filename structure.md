# Structure of the project
```
/TSPytorch
│
│
├── models
│   ├── __init__.py
│   ├── Informer.py
│   ├── Autoformer.py
│   ├── FEDformer.py
│   ├── ...
│   └── New_Model.py
│
├── utils
│   ├── __init__.py
│   ├── ...
│
│
├── layers
│   ├── __init__.py
│   ├── AutoCorrelation.py
│   ├── Transformer_EncDec.py
│   ├── utils.py
│   └── ...
│
├── dataset
│   ├── ETT
│   │   ├── ETTh1.csv
│   │   ├── ...
│   ├── exchange_rate.csv
│   └── weather.csv
│
├── scripts # 2 types: script to experiments and ablation
│   ├── experiments  
│   └── ablation
│
├── experiments
│   ├── model_saved
│   │   ├── checkpoints # model checkpoints:  *.pth, log.txt
│   │   │       ├── model1
│   │   │       ├── ...
│   │   └── results # model evaluation results: metrics, pred.npy
│   │       ├── model1
│   │       ├── ...
│   └── test_results (plot the test results)
│
│
│
├── experiment.py # script to run experiments -> scripts/experiments/
│
├── README.md
├── requirements.txt
└── Makefile

```








